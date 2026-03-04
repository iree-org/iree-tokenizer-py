// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Forward declarations — each file registers its bindings.
void RegisterTokenizer(nb::module_& m);
void RegisterStreaming(nb::module_& m);
void RegisterEncoding(nb::module_& m);

NB_MODULE(_iree_tokenizer, m) {
  m.doc() = "IREE Tokenizer — high-performance tokenizer bindings";
  RegisterEncoding(m);
  RegisterStreaming(m);
  RegisterTokenizer(m);
}
