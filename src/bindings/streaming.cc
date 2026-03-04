// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "streaming.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void RegisterStreaming(nb::module_& m) {
  nb::class_<EncodeStream>(m, "EncodeStream",
                           "Streaming encoder for incremental tokenization.")
      .def("feed", &EncodeStream::Feed, nb::arg("text"),
           "Feed a text chunk. Returns token IDs produced so far.")
      .def("finalize", &EncodeStream::Finalize,
           "Flush remaining tokens. Must be called after all input is fed.")
      .def("close", &EncodeStream::Close, "Release resources.")
      .def_prop_ro("is_open", &EncodeStream::is_open)
      .def("__enter__", [](nb::object self) -> nb::object { return self; })
      .def("__exit__", [](EncodeStream& s, nb::args) { s.Close(); });

  nb::class_<DecodeStream>(m, "DecodeStream",
                           "Streaming decoder for incremental detokenization.")
      .def("feed", &DecodeStream::Feed, nb::arg("tokens"),
           "Feed token IDs. Returns text produced so far.")
      .def("finalize", &DecodeStream::Finalize,
           "Flush remaining text. Must be called after all tokens are fed.")
      .def("close", &DecodeStream::Close, "Release resources.")
      .def_prop_ro("is_open", &DecodeStream::is_open)
      .def("__enter__", [](nb::object self) -> nb::object { return self; })
      .def("__exit__", [](DecodeStream& s, nb::args) { s.Close(); });
}
