# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""iree.tokenizer — High-performance tokenizer backed by IREE."""

from iree.tokenizer._iree_tokenizer import (
    Encoding,
    EncodeStream,
    DecodeStream,
    Tokenizer,
)

__all__ = [
    "Encoding",
    "EncodeStream",
    "DecodeStream",
    "Tokenizer",
    "decode_stream_iter",
]


def decode_stream_iter(tokenizer, token_iter, *, skip_special_tokens=False):
    """Iterate over decoded text chunks from a token iterator.

    Yields text chunks as they become available from the streaming decoder.
    This is the natural pattern for LLM inference token-at-a-time decoding.

    Args:
        tokenizer: A Tokenizer instance.
        token_iter: An iterable of token IDs (int).
        skip_special_tokens: If True, omit special tokens from output.

    Yields:
        str: Text chunks as they are decoded.
    """
    with tokenizer.decode_stream(skip_special_tokens=skip_special_tokens) as stream:
        for token in token_iter:
            tokens = [token] if isinstance(token, int) else list(token)
            text = stream.feed(tokens)
            if text:
                yield text
        text = stream.finalize()
        if text:
            yield text
