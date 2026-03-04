# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from iree.tokenizer import Tokenizer, decode_stream_iter


def test_encode_stream_basic(bpe_tokenizer):
    with bpe_tokenizer.encode_stream() as s:
        assert s.is_open
        all_ids = []
        all_ids.extend(s.feed("Hello "))
        all_ids.extend(s.feed("world"))
        all_ids.extend(s.finalize())
    assert not s.is_open
    # Should produce same tokens as one-shot.
    expected = bpe_tokenizer.encode("Hello world")
    assert all_ids == expected


def test_decode_stream_basic(bpe_tokenizer):
    ids = bpe_tokenizer.encode("Hello world")
    with bpe_tokenizer.decode_stream() as s:
        assert s.is_open
        text = ""
        # Feed one token at a time.
        for token_id in ids:
            text += s.feed([token_id])
        text += s.finalize()
    assert not s.is_open
    assert text == "Hello world"


def test_encode_stream_close(bpe_tokenizer):
    s = bpe_tokenizer.encode_stream()
    assert s.is_open
    s.close()
    assert not s.is_open
    with pytest.raises(RuntimeError, match="closed"):
        s.feed("text")


def test_decode_stream_close(bpe_tokenizer):
    s = bpe_tokenizer.decode_stream()
    assert s.is_open
    s.close()
    assert not s.is_open
    with pytest.raises(RuntimeError, match="closed"):
        s.feed([1])


def test_decode_stream_iter(bpe_tokenizer):
    ids = bpe_tokenizer.encode("Hello world")
    chunks = list(decode_stream_iter(bpe_tokenizer, ids))
    assert "".join(chunks) == "Hello world"


def test_decode_stream_iter_empty(bpe_tokenizer):
    chunks = list(decode_stream_iter(bpe_tokenizer, []))
    assert chunks == []


def test_context_manager_double_close(bpe_tokenizer):
    """Closing an already-closed stream should not crash."""
    with bpe_tokenizer.encode_stream() as s:
        s.close()
    # __exit__ calls close() again — should be safe.
