// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- TvecDialect.td - Tvec dialect definition ------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR `tvec` (tiled-vector) dialect: the machine-independent
// vectorized form that connects the virtual tile level (`pto`) to the hardware
// ISA level (`vpto`). See docs/designs/tile-to-hardware-dialect-layering.md.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TVEC_IR_TVECDIALECT
#define MLIR_DIALECT_TVEC_IR_TVECDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tvec Dialect
//===----------------------------------------------------------------------===//

def Tvec_Dialect : Dialect {
  let summary = "Machine-independent tiled-vector dialect (tile <-> hardware bridge).";
  let description = [{
    The `tvec` dialect models a tile operation that has already been turned
    into an explicit `scf.for` loop nest, but whose operations are still
    *hardware neutral*: values are logical vectors (`!tvec.vec<Nx T>`) and
    logical predicates (`!tvec.mask<T>`) with no physical register width, no
    `dist`/`part` distribution mode, no pipeline/sync semantics, and no
    `vecscope`.

    It sits between:
      * `pto`  - the virtual tile level (`pto.tadd` on `!pto.tile_buf`), and
      * `vpto` - the Da Vinci/Ascend hardware ISA (`vpto.vlds` on
                 `!vpto.vreg<64xf32>`).

    `tvec` is the level at which target-independent loop transformations run -
    most importantly **loop fusion** of sibling `scf.for` nests - before the
    program is committed to a physical register width and the concrete ISA.
  }];

  let name = "tvec";
  let cppNamespace = "::mlir::tvec";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLIR_DIALECT_TVEC_IR_TVECDIALECT
