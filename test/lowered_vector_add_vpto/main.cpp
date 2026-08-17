// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace PtoTestCommon;

void LaunchVectorAddKernel(uint8_t *workspace0, uint8_t *workspace1, float *input0,
                           float *input1, float *output, void *stream);

namespace {

constexpr size_t kElemCount = 256;
constexpr size_t kDataBytes = kElemCount * sizeof(float);
constexpr size_t kWorkspaceBytes = 1;

bool CheckAcl(aclError err, const char *what) {
    if (err == ACL_SUCCESS) {
        return true;
    }
    std::fprintf(stderr, "[ERROR] %s failed: %d\n", what, static_cast<int>(err));
    return false;
}

} // namespace

int main() {
    int rc = 0;
    int deviceId = 0;
    if (const char *envDevice = std::getenv("ACL_DEVICE_ID")) {
        deviceId = std::atoi(envDevice);
    }

    float *input0Host = nullptr;
    float *input1Host = nullptr;
    float *outputHost = nullptr;
    float *input0Device = nullptr;
    float *input1Device = nullptr;
    float *outputDevice = nullptr;
    uint8_t *workspace0Device = nullptr;
    uint8_t *workspace1Device = nullptr;
    aclrtStream stream = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&input0Host), kDataBytes),
                  "aclrtMallocHost(input0)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&input1Host), kDataBytes),
                  "aclrtMallocHost(input1)") ||
        !CheckAcl(aclrtMallocHost(reinterpret_cast<void **>(&outputHost), kDataBytes),
                  "aclrtMallocHost(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&input0Device), kDataBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(input0)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&input1Device), kDataBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(input1)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&outputDevice), kDataBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(output)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace0Device), kWorkspaceBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace0)") ||
        !CheckAcl(aclrtMalloc(reinterpret_cast<void **>(&workspace1Device), kWorkspaceBytes,
                              ACL_MEM_MALLOC_HUGE_FIRST),
                  "aclrtMalloc(workspace1)")) {
        rc = 1;
        goto cleanup;
    }

    {
        size_t input0Size = kDataBytes;
        size_t input1Size = kDataBytes;
        if (!ReadFile("input0.bin", input0Size, input0Host, kDataBytes) ||
            !ReadFile("input1.bin", input1Size, input1Host, kDataBytes)) {
            std::fprintf(stderr, "[ERROR] failed to read input0.bin or input1.bin\n");
            rc = 1;
            goto cleanup;
        }
    }

    if (!CheckAcl(aclrtMemcpy(input0Device, kDataBytes, input0Host, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(input0 H2D)") ||
        !CheckAcl(aclrtMemcpy(input1Device, kDataBytes, input1Host, kDataBytes,
                              ACL_MEMCPY_HOST_TO_DEVICE),
                  "aclrtMemcpy(input1 H2D)")) {
        rc = 1;
        goto cleanup;
    }

    LaunchVectorAddKernel(workspace0Device, workspace1Device, input0Device, input1Device,
                          outputDevice, stream);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CheckAcl(aclrtMemcpy(outputHost, kDataBytes, outputDevice, kDataBytes,
                              ACL_MEMCPY_DEVICE_TO_HOST),
                  "aclrtMemcpy(output D2H)")) {
        rc = 1;
        goto cleanup;
    }

    if (!WriteFile("output.bin", outputHost, kDataBytes)) {
        std::fprintf(stderr, "[ERROR] failed to write output.bin\n");
        rc = 1;
        goto cleanup;
    }

    std::printf("[INFO] wrote output.bin\n");

cleanup:
    if (workspace0Device != nullptr) aclrtFree(workspace0Device);
    if (workspace1Device != nullptr) aclrtFree(workspace1Device);
    if (input0Device != nullptr) aclrtFree(input0Device);
    if (input1Device != nullptr) aclrtFree(input1Device);
    if (outputDevice != nullptr) aclrtFree(outputDevice);
    if (input0Host != nullptr) aclrtFreeHost(input0Host);
    if (input1Host != nullptr) aclrtFreeHost(input1Host);
    if (outputHost != nullptr) aclrtFreeHost(outputHost);
    if (stream != nullptr) aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return rc;
}
