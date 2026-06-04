# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================
# tvec dialect TableGen targets.
#
# NOTE: This directory is intentionally NOT yet referenced from
# include/CMakeLists.txt (no `add_subdirectory(Tvec)`), so the existing build
# is unaffected. To enable the dialect, add that line and wire a TvecDialect
# library under lib/, per docs/designs/tile-to-hardware-dialect-layering.md §7.
# ============================================================

# Dialect + Operations (TvecOps.td transitively includes the dialect/types).
set(LLVM_TARGET_DEFINITIONS TvecOps.td)
mlir_tablegen(TvecDialect.h.inc -gen-dialect-decls -dialect=tvec)
mlir_tablegen(TvecDialect.cpp.inc -gen-dialect-defs -dialect=tvec)
mlir_tablegen(TvecOps.h.inc -gen-op-decls)
mlir_tablegen(TvecOps.cpp.inc -gen-op-defs)

# Types.
set(LLVM_TARGET_DEFINITIONS TvecTypes.td)
mlir_tablegen(TvecTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=tvec)
mlir_tablegen(TvecTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=tvec)

# Op interfaces.
set(LLVM_TARGET_DEFINITIONS TvecInterfaces.td)
mlir_tablegen(TvecInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(TvecInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(TvecOpsIncGen)
