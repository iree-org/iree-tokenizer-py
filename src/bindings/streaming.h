// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_PY_STREAMING_H_
#define IREE_TOKENIZER_PY_STREAMING_H_

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "iree/tokenizer/tokenizer.h"
#include "status_util.h"

// ---------------------------------------------------------------------------
// EncodeStream
// ---------------------------------------------------------------------------

class EncodeStream {
 public:
  EncodeStream(const iree_tokenizer_t* tokenizer, bool add_special_tokens)
      : tokenizer_(tokenizer) {
    iree_tokenizer_encode_flags_t flags =
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START;
    if (add_special_tokens) {
      flags |= IREE_TOKENIZER_ENCODE_FLAG_ADD_SPECIAL_TOKENS;
    }
    flags_ = flags;

    iree_host_size_t state_size = 0;
    iree::tokenizer::python::CheckStatus(
        iree_tokenizer_encode_state_calculate_size(tokenizer_, &state_size));

    state_storage_.resize(state_size);
    iree_host_size_t tb_size =
        iree_tokenizer_transform_buffer_recommended_size(8192);
    transform_buffer_.resize(tb_size);

    iree::tokenizer::python::CheckStatus(iree_tokenizer_encode_state_initialize(
        tokenizer_,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        iree_make_byte_span(transform_buffer_.data(), transform_buffer_.size()),
        iree_tokenizer_offset_run_list_empty(), flags_, &state_));
  }

  ~EncodeStream() { Close(); }

  std::vector<int32_t> Feed(const std::string& text) {
    EnsureOpen();
    std::vector<int32_t> all_tokens;
    constexpr size_t kBatchSize = 1024;
    std::vector<iree_tokenizer_token_id_t> token_buf(kBatchSize);
    iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
        token_buf.data(), nullptr, nullptr, kBatchSize);

    iree_host_size_t offset = 0;
    while (offset < text.size()) {
      iree_string_view_t chunk = {.data = text.data() + offset,
                                  .size = text.size() - offset};
      iree_host_size_t bytes_consumed = 0;
      iree_host_size_t token_count = 0;
      iree::tokenizer::python::CheckStatus(iree_tokenizer_encode_state_feed(
          state_, chunk, output, &bytes_consumed, &token_count));
      all_tokens.insert(all_tokens.end(), token_buf.data(),
                        token_buf.data() + token_count);
      offset += bytes_consumed;
      if (bytes_consumed == 0 && token_count == 0) break;
    }
    return all_tokens;
  }

  std::vector<int32_t> Finalize() {
    EnsureOpen();
    constexpr size_t kBatchSize = 256;
    std::vector<iree_tokenizer_token_id_t> token_buf(kBatchSize);
    iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
        token_buf.data(), nullptr, nullptr, kBatchSize);
    iree_host_size_t token_count = 0;
    iree::tokenizer::python::CheckStatus(
        iree_tokenizer_encode_state_finalize(state_, output, &token_count));
    return std::vector<int32_t>(token_buf.data(),
                                token_buf.data() + token_count);
  }

  void Close() {
    if (state_) {
      iree_tokenizer_encode_state_deinitialize(state_);
      state_ = nullptr;
    }
  }

  bool is_open() const { return state_ != nullptr; }

 private:
  void EnsureOpen() {
    if (!state_) throw std::runtime_error("EncodeStream is closed");
  }

  const iree_tokenizer_t* tokenizer_;
  iree_tokenizer_encode_flags_t flags_;
  std::vector<uint8_t> state_storage_;
  std::vector<uint8_t> transform_buffer_;
  iree_tokenizer_encode_state_t* state_ = nullptr;
};

// ---------------------------------------------------------------------------
// DecodeStream
// ---------------------------------------------------------------------------

class DecodeStream {
 public:
  DecodeStream(const iree_tokenizer_t* tokenizer, bool skip_special_tokens)
      : tokenizer_(tokenizer) {
    iree_tokenizer_decode_flags_t flags = IREE_TOKENIZER_DECODE_FLAG_NONE;
    if (skip_special_tokens) {
      flags |= IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS;
    }

    iree_host_size_t state_size = 0;
    iree::tokenizer::python::CheckStatus(
        iree_tokenizer_decode_state_calculate_size(tokenizer_, &state_size));
    state_storage_.resize(state_size);

    iree::tokenizer::python::CheckStatus(iree_tokenizer_decode_state_initialize(
        tokenizer_, flags,
        iree_make_byte_span(state_storage_.data(), state_storage_.size()),
        &state_));
  }

  ~DecodeStream() { Close(); }

  std::string Feed(const std::vector<int32_t>& tokens) {
    EnsureOpen();
    std::string all_text;
    std::vector<char> text_buf(IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE);

    iree_host_size_t offset = 0;
    while (offset < tokens.size()) {
      iree_tokenizer_token_id_list_t list = {.count = tokens.size() - offset,
                                             .values = tokens.data() + offset};
      iree_mutable_string_view_t text_output = {.data = text_buf.data(),
                                                .size = text_buf.size()};
      iree_host_size_t tokens_consumed = 0;
      iree_host_size_t text_length = 0;
      iree::tokenizer::python::CheckStatus(iree_tokenizer_decode_state_feed(
          state_, list, text_output, &tokens_consumed, &text_length));
      all_text.append(text_buf.data(), text_length);
      offset += tokens_consumed;
      if (tokens_consumed == 0 && text_length == 0) break;
    }
    return all_text;
  }

  std::string Finalize() {
    EnsureOpen();
    std::vector<char> text_buf(IREE_TOKENIZER_DECODE_OUTPUT_RECOMMENDED_SIZE);
    iree_mutable_string_view_t text_output = {.data = text_buf.data(),
                                              .size = text_buf.size()};
    iree_host_size_t text_length = 0;
    iree::tokenizer::python::CheckStatus(iree_tokenizer_decode_state_finalize(
        state_, text_output, &text_length));
    return std::string(text_buf.data(), text_length);
  }

  void Close() {
    if (state_) {
      iree_tokenizer_decode_state_deinitialize(state_);
      state_ = nullptr;
    }
  }

  bool is_open() const { return state_ != nullptr; }

 private:
  void EnsureOpen() {
    if (!state_) throw std::runtime_error("DecodeStream is closed");
  }

  const iree_tokenizer_t* tokenizer_;
  std::vector<uint8_t> state_storage_;
  iree_tokenizer_decode_state_t* state_ = nullptr;
};

#endif  // IREE_TOKENIZER_PY_STREAMING_H_
