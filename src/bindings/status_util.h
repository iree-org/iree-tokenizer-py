// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_PY_STATUS_UTIL_H_
#define IREE_TOKENIZER_PY_STATUS_UTIL_H_

#include <nanobind/nanobind.h>

#include <string>

#include "iree/base/api.h"

namespace iree::tokenizer::python {

// Raises a Python exception from an iree_status_t and frees the status.
// The status must not be OK.
[[noreturn]] inline void RaiseFromStatus(iree_status_t status) {
  iree_status_code_t code = iree_status_code(status);

  // Extract message via allocator-based API.
  char* buf = nullptr;
  iree_host_size_t buf_len = 0;
  iree_allocator_t alloc = iree_allocator_system();
  std::string msg;
  if (iree_status_to_string(status, &alloc, &buf, &buf_len)) {
    msg.assign(buf, buf_len);
    iree_allocator_free(alloc, buf);
  } else {
    msg = iree_status_code_string(code);
  }
  iree_status_free(status);

  switch (code) {
    case IREE_STATUS_INVALID_ARGUMENT:
      throw nanobind::value_error(msg.c_str());
    case IREE_STATUS_NOT_FOUND:
      throw nanobind::key_error(msg.c_str());
    case IREE_STATUS_UNIMPLEMENTED:
      throw nanobind::type_error(msg.c_str());
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      PyErr_SetString(PyExc_MemoryError, msg.c_str());
      throw nanobind::python_error();
    default:
      throw std::runtime_error(msg);
  }
}

// Check status and raise if not OK.
inline void CheckStatus(iree_status_t status) {
  if (IREE_LIKELY(iree_status_is_ok(status))) return;
  RaiseFromStatus(status);
}

}  // namespace iree::tokenizer::python

#endif  // IREE_TOKENIZER_PY_STATUS_UTIL_H_
