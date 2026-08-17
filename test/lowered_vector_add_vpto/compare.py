#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# coding=utf-8

import sys
import numpy as np

ELEM_COUNT = 256
EPS = 1e-6

golden = np.fromfile("golden.bin", dtype=np.float32)
output = np.fromfile("output.bin", dtype=np.float32)

if golden.size != ELEM_COUNT or output.size != ELEM_COUNT:
    print(f"[ERROR] shape mismatch: golden={golden.size}, output={output.size}")
    sys.exit(2)

if np.allclose(golden, output, atol=EPS, rtol=EPS, equal_nan=True):
    print("[INFO] compare passed")
    sys.exit(0)

abs_diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
idx = int(np.argmax(abs_diff))
print(
    f"[ERROR] compare failed: max diff={float(abs_diff[idx])} at idx={idx} "
    f"golden={float(golden[idx])} output={float(output[idx])}"
)
sys.exit(2)
