# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from iree.tokenizer import Tokenizer


def test_decode_basic(bpe_tokenizer):
    ids = bpe_tokenizer.encode("Hello world")
    text = bpe_tokenizer.decode(ids)
    assert text == "Hello world"


def test_decode_empty(bpe_tokenizer):
    text = bpe_tokenizer.decode([])
    assert text == ""


def test_decode_roundtrip_unicode(bpe_tokenizer):
    # The minimal BPE tokenizer may not handle all unicode, but ASCII should
    # round-trip.
    for text in ["Hello", "abc 123", "foo bar baz"]:
        ids = bpe_tokenizer.encode(text)
        assert bpe_tokenizer.decode(ids) == text
