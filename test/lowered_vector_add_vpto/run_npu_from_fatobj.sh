#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build-npu}"
RUN_DIR="${RUN_DIR:-${BUILD_DIR}/run}"
KERNEL_OBJ="${KERNEL_OBJ:-${SCRIPT_DIR}/vector_add_kernel.o}"
ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
BISHENG_BIN="${BISHENG_BIN:-bisheng}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the CANN environment first." >&2
    exit 1
fi
if [[ ! -f "${KERNEL_OBJ}" ]]; then
    echo "[ERROR] prebuilt fatobj not found: ${KERNEL_OBJ}" >&2
    echo "[ERROR] Copy vector_add_kernel.o here or set KERNEL_OBJ=/path/to/vector_add_kernel.o." >&2
    exit 1
fi
if ! command -v "${BISHENG_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] bisheng not found. Source the CANN environment first or set BISHENG_BIN." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${RUN_DIR}"

"${BISHENG_BIN}" \
    -c -fPIC -xcce \
    -fPIC -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-arch=dav-c310-vec \
    -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes -Wno-unknown-attributes \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/pkg_inc" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    "${SCRIPT_DIR}/launch.cpp" \
    -o "${BUILD_DIR}/launch.o"

"${BISHENG_BIN}" \
    -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
    -shared -Wl,-soname,libvector_add_kernel.so \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${BUILD_DIR}/libvector_add_kernel.so" \
    "${KERNEL_OBJ}" \
    "${BUILD_DIR}/launch.o" \
    -Wl,--no-as-needed -lruntime

"${BISHENG_BIN}" \
    -xc++ -include stdint.h -include stddef.h -std=c++17 \
    "${SCRIPT_DIR}/main.cpp" \
    -I "${SCRIPT_DIR}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_DRIVER_PATH}/kernel/inc" \
    -L "${BUILD_DIR}" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${BUILD_DIR}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${BUILD_DIR}/lowered_vector_add_vpto" \
    -lvector_add_kernel \
    -Wl,--allow-shlib-undefined -lruntime \
    -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase -lpthread

cd "${RUN_DIR}"
python3 "${SCRIPT_DIR}/gen_data.py"
LD_LIBRARY_PATH="${BUILD_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
    "${BUILD_DIR}/lowered_vector_add_vpto"
python3 "${SCRIPT_DIR}/compare.py"
