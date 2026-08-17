# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
set(CMAKE_COMPILER bisheng)
set(CMAKE_C_COMPILER ${CMAKE_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_COMPILER})
project(lowered_vector_add_vpto)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
    set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH is not set. Source the CANN environment first.")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ASCEND_DRIVER_PATH /usr/local/Ascend/driver)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION Ascend950PR_9599)
endif()

if(NOT DEFINED PTOAS_BIN)
    find_program(PTOAS_BIN ptoas)
endif()
if(NOT PTOAS_BIN)
    message(FATAL_ERROR "PTOAS_BIN is not set and ptoas was not found in PATH.")
endif()

if(DEFINED ENV{SIM_LIB_DIR})
    set(SIM_LIB_DIR $ENV{SIM_LIB_DIR})
else()
    set(SIM_LIB_DIR ${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib)
endif()

add_compile_options(
    -D_FORTIFY_SOURCE=2
    -O2
    -std=c++17
    -Wno-macro-redefined
    -Wno-ignored-attributes
    -Wno-unknown-attributes
    -fstack-protector-strong
    -fPIC
)
add_link_options(
    -s
    -Wl,-z,relro
    -Wl,-z,now
)

set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -fPIC
    -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
)

set(CMAKE_CPP_COMPILE_OPTIONS
    -xc++
    "SHELL:-include stdint.h"
    "SHELL:-include stddef.h"
)

include_directories(
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_DRIVER_PATH}/kernel/inc
)

set(KERNEL_MLIR ${CMAKE_CURRENT_SOURCE_DIR}/lowered_vector_add_kernel_vpto.mlir)
set(KERNEL_FATOBJ ${CMAKE_CURRENT_BINARY_DIR}/vector_add_kernel.o)

add_custom_command(
    OUTPUT ${KERNEL_FATOBJ}
    COMMAND ${PTOAS_BIN}
        --pto-arch=a5
        --pto-backend=vpto
        ${KERNEL_MLIR}
        -o
        ${KERNEL_FATOBJ}
    DEPENDS ${KERNEL_MLIR}
    COMMENT "ptoas: lowered VPTO MLIR -> vector_add_kernel.o"
    VERBATIM
)

add_library(vector_add_kernel SHARED launch.cpp ${KERNEL_FATOBJ})
set_source_files_properties(${KERNEL_FATOBJ}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
target_compile_options(vector_add_kernel PRIVATE
    ${CMAKE_CCE_COMPILE_OPTIONS}
    --cce-aicore-arch=dav-c310-vec
    -std=c++17
)
target_include_directories(vector_add_kernel PRIVATE
    ${ASCEND_HOME_PATH}/pkg_inc/
    ${ASCEND_HOME_PATH}/pkg_inc/profiling/
    ${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime
)
target_link_options(vector_add_kernel PRIVATE --cce-fatobj-link)

add_executable(lowered_vector_add_vpto main.cpp)
target_compile_options(lowered_vector_add_vpto PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS})
target_include_directories(lowered_vector_add_vpto PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
)
target_link_directories(lowered_vector_add_vpto PUBLIC
    ${ASCEND_HOME_PATH}/lib64
    ${SIM_LIB_DIR}
)
target_link_libraries(lowered_vector_add_vpto PRIVATE
    vector_add_kernel
    runtime_camodel
    stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread
)
