// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- TvecInterfaces.td ------------------------------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Op interfaces that make `tvec-fuse-loops` cheap: they expose the tile
// read/write footprint and the per-lane/pure nature of the ops without
// hard-coding op names, so the loop-fusion pass can build dependence sets and
// forward intermediate vectors generically.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TVEC_IR_TVECINTERFACES
#define MLIR_DIALECT_TVEC_IR_TVECINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TvecTileAccessOpInterface
//===----------------------------------------------------------------------===//

def Tvec_TileAccessOpInterface : OpInterface<"TileAccessOpInterface"> {
  let description = [{
    Implemented by the memory ops (`tvec.load`, `tvec.store`). Exposes the tile
    (memref) being touched, the row/col strip subscripts, and the direction of
    the access. `tvec-fuse-loops` uses this to compute the read/write sets of
    two candidate loop bodies and prove they are independent (apart from the
    single producer->consumer value being forwarded).
  }];
  let cppNamespace = "::mlir::tvec";
  let methods = [
    InterfaceMethod<"Return the tile memref being accessed.",
      "::mlir::Value", "getTile">,
    InterfaceMethod<"Return the row index SSA value.",
      "::mlir::Value", "getRowIndex">,
    InterfaceMethod<"Return the column index SSA value.",
      "::mlir::Value", "getColIndex">,
    InterfaceMethod<"Return the predicate mask operand.",
      "::mlir::Value", "getMask">,
    InterfaceMethod<"Return true if this access writes the tile (store).",
      "bool", "isWrite">
  ];
}

//===----------------------------------------------------------------------===//
// TvecElementwiseOpInterface
//===----------------------------------------------------------------------===//

def Tvec_ElementwiseOpInterface : OpInterface<"ElementwiseOpInterface"> {
  let description = [{
    Implemented by the pure, per-lane compute ops (`tvec.add`, `tvec.mul`,
    `tvec.convert`, ...). Marks the op as safe to sink/merge across a fused loop
    boundary and exposes its mask so the fuser can check predicate
    compatibility between the producer and consumer bodies.
  }];
  let cppNamespace = "::mlir::tvec";
  let methods = [
    InterfaceMethod<"Return the predicate mask operand, or null if unmasked.",
      "::mlir::Value", "getMaskOrNull", (ins), [{}],
      /*defaultImplementation=*/[{ return ::mlir::Value(); }]>,
    InterfaceMethod<"Return true: the op is pure and computes lane-by-lane.",
      "bool", "isPerLanePure", (ins), [{}],
      /*defaultImplementation=*/[{ return true; }]>
  ];
}

#endif // MLIR_DIALECT_TVEC_IR_TVECINTERFACES
