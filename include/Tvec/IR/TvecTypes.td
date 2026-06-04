// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- TvecTypes.td -----------------------------------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Types for the `tvec` dialect. Both types are *logical*: they carry an element
// type and (for vectors) a lane count that is a tiling hint chosen by the
// producing template, NOT the physical hardware register width. The
// `tvec-select-width` pass re-tiles these before lowering to `vpto`, where the
// physical `!vpto.vreg<64xf32>` / `!vpto.mask<b32>` types take over.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TVEC_IR_TVECTYPES
#define MLIR_DIALECT_TVEC_IR_TVECTYPES

include "mlir/IR/AttrTypeBase.td"
include "Tvec/IR/TvecDialect.td"

class Tvec_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tvec_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// VecType : !tvec.vec<Nx T>
//===----------------------------------------------------------------------===//

def Tvec_VecType : Tvec_Type<"Vec", "vec"> {
  let summary = "A logical, machine-independent vector strip.";
  let description = [{
    `!tvec.vec<64xf32>` is a strip of `elementCount` lanes of `elementType`.
    `elementCount` is a logical tiling hint, not a physical register width: it
    is chosen by the expanding template and re-tiled by `tvec-select-width`
    before lowering to the hardware `!vpto.vreg`.
  }];

  let parameters = (ins
    "int64_t":$elementCount,
    "::mlir::Type":$elementType
  );

  let assemblyFormat = "`<` $elementCount `x` $elementType `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// MaskType : !tvec.mask<T>
//===----------------------------------------------------------------------===//

def Tvec_MaskType : Tvec_Type<"Mask", "mask"> {
  let summary = "A logical per-lane predicate.";
  let description = [{
    `!tvec.mask<f32>` is a machine-independent predicate that selects active
    lanes for masked `tvec` ops (tail predication via `tvec.make_mask`). It
    carries only its element type; the granularity-encoded hardware predicate
    `!vpto.mask<b32>` is produced during lowering.
  }];

  let parameters = (ins "::mlir::Type":$elementType);

  let assemblyFormat = "`<` $elementType `>`";
}

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

def Tvec_AnyVec : Type<CPred<"::llvm::isa<::mlir::tvec::VecType>($_self)">,
                       "tvec logical vector">;
def Tvec_AnyMask : Type<CPred<"::llvm::isa<::mlir::tvec::MaskType>($_self)">,
                        "tvec logical mask">;

#endif // MLIR_DIALECT_TVEC_IR_TVECTYPES
