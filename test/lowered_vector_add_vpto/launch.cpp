// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void vector_add_kernel(__gm__ uint8_t *workspace0,
                                                     __gm__ uint8_t *workspace1,
                                                     __gm__ float *input0,
                                                     __gm__ float *input1,
                                                     __gm__ float *output,
                                                     int32_t scalar0,
                                                     int32_t scalar1,
                                                     int32_t scalar2);

void LaunchVectorAddKernel(uint8_t *workspace0, uint8_t *workspace1, float *input0,
                           float *input1, float *output, void *stream) {
    vector_add_kernel<<<1, nullptr, stream>>>(
        (__gm__ uint8_t *)workspace0,
        (__gm__ uint8_t *)workspace1,
        (__gm__ float *)input0,
        (__gm__ float *)input1,
        (__gm__ float *)output,
        0,
        0,
        0);
}
