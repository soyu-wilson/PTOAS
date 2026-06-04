// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- TvecOps.td - Tvec dialect operations ----------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Operations of the machine-independent tiled-vector (`tvec`) dialect. The set
// is a hardware-neutral counterpart of the `pto.v*` ISA family used by the
// TileLang templates, minus everything hardware-specific (dist modes,
// granularity-typed masks, vecscope, pipes/sync).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TVEC_IR_TVECOPS
#define MLIR_DIALECT_TVEC_IR_TVECOPS

include "Tvec/IR/TvecDialect.td"
include "Tvec/IR/TvecTypes.td"
include "Tvec/IR/TvecInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class Tvec_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tvec_Dialect, mnemonic, traits>;

def Tvec_AnyVecMemRef :
  Type<CPred<"::llvm::isa<::mlir::MemRefType>($_self)">,
       "tile-backing memref">;

//===----------------------------------------------------------------------===//
// Memory ops (the only side-effecting ops; implement TileAccessOpInterface)
//===----------------------------------------------------------------------===//

def Tvec_LoadOp : Tvec_Op<"load", [
    DeclareOpInterfaceMethods<Tvec_TileAccessOpInterface>,
    MemoryEffects<[MemRead]>]> {
  let summary = "Load one masked vector strip from a tile memref.";
  let description = [{
    Reads the `[row, col:col+N)` strip of `tile` under `mask` into a logical
    vector. Hardware-neutral counterpart of `pto.vlds` (no `dist` mode).
  }];
  let arguments = (ins Tvec_AnyVecMemRef:$tile,
                       Index:$row,
                       Index:$col,
                       Tvec_AnyMask:$mask);
  let results = (outs Tvec_AnyVec:$result);
  let assemblyFormat = [{
    $tile `[` $row `,` $col `]` `,` $mask attr-dict
      `:` type($tile) `,` type($mask) `->` type($result)
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getTile() { return getODSOperands(0)[0]; }
    ::mlir::Value getRowIndex() { return getRow(); }
    ::mlir::Value getColIndex() { return getCol(); }
    bool isWrite() { return false; }
  }];
}

def Tvec_StoreOp : Tvec_Op<"store", [
    DeclareOpInterfaceMethods<Tvec_TileAccessOpInterface>,
    MemoryEffects<[MemWrite]>]> {
  let summary = "Store one masked vector strip to a tile memref.";
  let description = [{
    Writes `value` to the `[row, col:col+N)` strip of `tile` under `mask`.
    Hardware-neutral counterpart of `pto.vsts` (no `dist` mode). Together with
    `tvec.load`, these are the only `tvec` ops with memory effects, so loop
    fusion legality reduces to dependence analysis over these accesses.
  }];
  let arguments = (ins Tvec_AnyVec:$value,
                       Tvec_AnyVecMemRef:$tile,
                       Index:$row,
                       Index:$col,
                       Tvec_AnyMask:$mask);
  let results = (outs);
  let assemblyFormat = [{
    $value `,` $tile `[` $row `,` $col `]` `,` $mask attr-dict
      `:` type($value) `,` type($tile) `,` type($mask)
  }];

  let extraClassDeclaration = [{
    ::mlir::Value getTile() { return getODSOperands(1)[0]; }
    ::mlir::Value getRowIndex() { return getRow(); }
    ::mlir::Value getColIndex() { return getCol(); }
    bool isWrite() { return true; }
  }];
}

//===----------------------------------------------------------------------===//
// Predication / broadcast
//===----------------------------------------------------------------------===//

def Tvec_MakeMaskOp : Tvec_Op<"make_mask", [Pure]> {
  let summary = "Build a tail-predication mask for the remaining lanes.";
  let description = [{
    Given the remaining valid element count, produce the active-lane `mask` and
    the updated remaining count. Hardware-neutral counterpart of the templates'
    `pto.make_mask`.
  }];
  let arguments = (ins Index:$remaining);
  let results = (outs Tvec_AnyMask:$mask, Index:$nextRemaining);
  let assemblyFormat = [{
    $remaining attr-dict `:` type($mask)
  }];
}

def Tvec_BroadcastOp : Tvec_Op<"broadcast", [Pure]> {
  let summary = "Splat a scalar across all lanes of a logical vector.";
  let arguments = (ins AnyType:$scalar);
  let results = (outs Tvec_AnyVec:$result);
  let assemblyFormat = [{
    $scalar attr-dict `:` type($scalar) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Elementwise compute ops (pure; implement ElementwiseOpInterface)
//===----------------------------------------------------------------------===//

class Tvec_BinaryOp<string mnemonic> : Tvec_Op<mnemonic, [
    Pure,
    DeclareOpInterfaceMethods<Tvec_ElementwiseOpInterface>,
    AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins Tvec_AnyVec:$lhs, Tvec_AnyVec:$rhs, Tvec_AnyMask:$mask);
  let results = (outs Tvec_AnyVec:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $mask attr-dict `:` type($lhs) `,` type($mask)
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getMaskOrNull() { return getMask(); }
  }];
}

def Tvec_AddOp : Tvec_BinaryOp<"add">;
def Tvec_SubOp : Tvec_BinaryOp<"sub">;
def Tvec_MulOp : Tvec_BinaryOp<"mul">;
def Tvec_DivOp : Tvec_BinaryOp<"div">;
def Tvec_MaxOp : Tvec_BinaryOp<"max">;
def Tvec_MinOp : Tvec_BinaryOp<"min">;
def Tvec_AndOp : Tvec_BinaryOp<"and">;
def Tvec_OrOp  : Tvec_BinaryOp<"or">;

def Tvec_ConvertOp : Tvec_Op<"convert", [
    Pure, DeclareOpInterfaceMethods<Tvec_ElementwiseOpInterface>]> {
  let summary = "Per-lane element-type conversion (counterpart of pto.vcvt).";
  let description = [{
    Converts a logical vector to a different element type. `result` may have a
    different lane count than `value` (e.g. f16<->f32 pack/unpack), which
    `tvec-select-width` and the `vpto` lowering resolve into the concrete
    `part`/`dist` encodings.
  }];
  let arguments = (ins Tvec_AnyVec:$value, Tvec_AnyMask:$mask);
  let results = (outs Tvec_AnyVec:$result);
  let assemblyFormat = [{
    $value `,` $mask attr-dict `:` type($value) `,` type($mask) `->` type($result)
  }];
  let extraClassDeclaration = [{
    ::mlir::Value getMaskOrNull() { return getMask(); }
  }];
}

#endif // MLIR_DIALECT_TVEC_IR_TVECOPS
