# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Security and robustness tests."""

import json
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Edge-case inputs
# ---------------------------------------------------------------------------


class TestEdgeCaseInputs:
    def test_encode_empty_string(self, bpe_tokenizer):
        ids = bpe_tokenizer.encode("")
        assert ids == []

    def test_decode_empty_list(self, bpe_tokenizer):
        text = bpe_tokenizer.decode([])
        assert text == ""

    def test_encode_batch_empty(self, bpe_tokenizer):
        result = bpe_tokenizer.encode_batch([])
        assert result == []

    def test_decode_batch_empty(self, bpe_tokenizer):
        result = bpe_tokenizer.decode_batch([])
        assert result == []

    def test_encode_null_bytes(self, bpe_tokenizer):
        """Null bytes in input should not crash."""
        ids = bpe_tokenizer.encode("Hello\x00world")
        assert isinstance(ids, list)

    def test_encode_unicode_edge_cases(self, bpe_tokenizer):
        """Various Unicode edge cases should not crash."""
        for text in ["\U0001f600", "a" * 100000, "\u00e9\u00e9\u00e9"]:
            try:
                bpe_tokenizer.encode(text)
            except (ValueError, TypeError):
                pass  # Rejecting is fine, crashing is not.

    def test_encode_very_long_input(self, bpe_tokenizer):
        """Large input should not crash."""
        ids = bpe_tokenizer.encode("x" * 50000)
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# Loading errors
# ---------------------------------------------------------------------------


class TestLoadErrors:
    def test_from_file_nonexistent(self):
        from iree.tokenizer import Tokenizer

        with pytest.raises(ValueError, match="Cannot open"):
            Tokenizer.from_file("/nonexistent/tokenizer.json")

    def test_from_str_empty(self):
        from iree.tokenizer import Tokenizer

        with pytest.raises(Exception):
            Tokenizer.from_str("")

    def test_from_str_malformed_json(self):
        from iree.tokenizer import Tokenizer

        with pytest.raises(Exception):
            Tokenizer.from_str("{invalid json")

    def test_from_buffer_empty(self):
        from iree.tokenizer import Tokenizer

        with pytest.raises(Exception):
            Tokenizer.from_buffer(b"")


# ---------------------------------------------------------------------------
# Streaming edge cases
# ---------------------------------------------------------------------------


class TestStreamingEdgeCases:
    def test_encode_stream_feed_after_finalize(self, bpe_tokenizer):
        """Feeding after finalize should not crash."""
        s = bpe_tokenizer.encode_stream()
        s.feed("Hello")
        s.finalize()
        # May raise or return empty — either is fine, just must not crash.
        try:
            s.feed("more")
        except RuntimeError:
            pass

    def test_decode_stream_feed_after_finalize(self, bpe_tokenizer):
        """Feeding after finalize should not crash."""
        s = bpe_tokenizer.decode_stream()
        s.feed([72])
        s.finalize()
        try:
            s.feed([73])
        except RuntimeError:
            pass

    def test_encode_stream_double_finalize(self, bpe_tokenizer):
        """Double finalize should not crash."""
        s = bpe_tokenizer.encode_stream()
        s.feed("Hello")
        s.finalize()
        try:
            s.finalize()
        except RuntimeError:
            pass

    def test_decode_stream_double_finalize(self, bpe_tokenizer):
        """Double finalize should not crash."""
        s = bpe_tokenizer.decode_stream()
        s.feed([72])
        s.finalize()
        try:
            s.finalize()
        except RuntimeError:
            pass

    def test_encode_stream_feed_empty(self, bpe_tokenizer):
        """Empty feed should be harmless."""
        with bpe_tokenizer.encode_stream() as s:
            ids = s.feed("")
            assert isinstance(ids, list)
            s.finalize()

    def test_decode_stream_feed_empty(self, bpe_tokenizer):
        """Empty feed should be harmless."""
        with bpe_tokenizer.decode_stream() as s:
            text = s.feed([])
            assert isinstance(text, str)
            s.finalize()


# ---------------------------------------------------------------------------
# CLI error handling
# ---------------------------------------------------------------------------


CLI = [sys.executable, "-m", "iree.tokenizer.cli"]


class TestCLIErrors:
    def test_missing_tokenizer(self):
        """CLI without -t should exit with error."""
        r = subprocess.run(
            [*CLI, "encode", "--no-progress"],
            input="Hello\n",
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0

    def test_nonexistent_tokenizer_file(self):
        """CLI with nonexistent tokenizer should exit with error."""
        r = subprocess.run(
            [*CLI, "encode", "-t", "/nonexistent/tok.json", "--no-progress"],
            input="Hello\n",
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0

    def test_decode_invalid_json_input(self):
        """Decode with non-JSON input should handle gracefully."""
        import pathlib

        tok_path = str(
            pathlib.Path(__file__).parent / "testdata" / "bpe_bytelevel_minimal.json"
        )
        r = subprocess.run(
            [*CLI, "decode", "-t", tok_path, "--no-progress"],
            input="not json\n",
            capture_output=True,
            text=True,
        )
        # Should fail but not crash (returncode != 0 or empty output).
        assert r.returncode != 0 or r.stdout.strip() == ""
