// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- TvecDialect.h - Tvec dialect C++ entry point ----------------------===//
//
// Aggregation header for the machine-independent tiled-vector (`tvec`) dialect.
// See docs/designs/tile-to-hardware-dialect-layering.md.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TVEC_IR_TVECDIALECT_H
#define MLIR_DIALECT_TVEC_IR_TVECDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Op interface declarations (generated from TvecInterfaces.td).
#include "Tvec/IR/TvecInterfaces.h.inc"

// Dialect declaration (generated from TvecDialect.td via TvecOps.td).
#include "Tvec/IR/TvecDialect.h.inc"

// Type declarations (generated from TvecTypes.td).
#define GET_TYPEDEF_CLASSES
#include "Tvec/IR/TvecTypeDefs.h.inc"

// Op declarations (generated from TvecOps.td).
#define GET_OP_CLASSES
#include "Tvec/IR/TvecOps.h.inc"

#endif // MLIR_DIALECT_TVEC_IR_TVECDIALECT_H
