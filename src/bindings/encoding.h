// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_PY_ENCODING_H_
#define IREE_TOKENIZER_PY_ENCODING_H_

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Rich encoding result with numpy arrays.
// Fields are set directly from C++ after default construction.
struct Encoding {
  nb::object ids;       // numpy int32 array
  nb::object offsets;   // numpy uint64 array (n, 2) or None
  nb::object type_ids;  // numpy uint8 array
};

#endif  // IREE_TOKENIZER_PY_ENCODING_H_
