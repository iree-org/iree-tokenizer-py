// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "encoding.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

void RegisterEncoding(nb::module_& m) {
  nb::class_<Encoding>(m, "Encoding",
                       "Rich encoding result with IDs, offsets, and type IDs.")
      .def(nb::init<>())
      .def_rw("ids", &Encoding::ids, "Token IDs as numpy int32 array.")
      .def_rw("offsets", &Encoding::offsets,
              "Byte offsets as numpy uint64 array of shape (n_tokens, 2), "
              "or None if offset tracking was not requested.")
      .def_rw("type_ids", &Encoding::type_ids,
              "Type/segment IDs as numpy uint8 array.")
      .def("__len__",
           [](const Encoding& e) -> size_t {
             if (e.ids.is_none()) return 0;
             return nb::cast<size_t>(e.ids.attr("__len__")());
           })
      .def("__repr__", [](const Encoding& e) {
        size_t n = 0;
        if (!e.ids.is_none()) {
          n = nb::cast<size_t>(e.ids.attr("__len__")());
        }
        return "Encoding(n_tokens=" + std::to_string(n) + ")";
      });
}
