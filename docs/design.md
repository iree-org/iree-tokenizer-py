# iree-tokenizer Design Reference

**Package**: `iree-tokenizer` (PyPI) / `import iree.tokenizer` (Python)
**Status**: v0.1 design
**Backend**: IREE tokenizer C library (BPE, WordPiece, Unigram)

## Overview

Python bindings for the IREE tokenizer via nanobind. Loads HuggingFace
`tokenizer.json` files and exposes encode/decode with batch and streaming
support. Designed for maximum performance in inference workloads.

**Non-goals for v1**: training, padding/truncation, `from_pretrained`
(network), async, custom Python normalizers/pre-tokenizers.

---

## Python API

### Loading

```python
import iree.tokenizer

# From file (most common)
tok = iree.tokenizer.Tokenizer.from_file("tokenizer.json")

# From JSON string
tok = iree.tokenizer.Tokenizer.from_str(json_string)

# From bytes
tok = iree.tokenizer.Tokenizer.from_buffer(json_bytes)
```

All constructors parse HuggingFace `tokenizer.json` format. The returned
`Tokenizer` is immutable and thread-safe.

### Encode

```python
# Returns list[int] — converges with HF encode().ids and tiktoken.encode()
ids: list[int] = tok.encode("hello world")
ids: list[int] = tok.encode("hello world", add_special_tokens=True)

# Returns numpy int32 array — like tiktoken.encode_to_numpy()
arr: np.ndarray = tok.encode_to_array("hello world")

# Rich output with offsets and type IDs
enc: Encoding = tok.encode_rich("hello world", track_offsets=True)
# enc.ids: np.ndarray[int32]
# enc.offsets: np.ndarray[uint64] shaped (n, 2) — byte ranges into original text
# enc.type_ids: np.ndarray[uint8] — segment IDs (0=seq_a, 1=seq_b)
```

Default `add_special_tokens=False` matches tiktoken's default. HF defaults to
`True` but their API is `encode(text).ids` which is richer.

### Decode

```python
text: str = tok.decode([1, 2, 3])
text: str = tok.decode([1, 2, 3], skip_special_tokens=True)
```

Accepts `list[int]`, `tuple[int, ...]`, or `np.ndarray`. Returns UTF-8 string.

### Batch

Uses the C batch API with shared state (single allocation, no thread pool).

```python
# List-of-lists
ids_list: list[list[int]] = tok.encode_batch(["hello", "world"])
ids_list: list[list[int]] = tok.encode_batch(texts, add_special_tokens=True)
texts: list[str] = tok.decode_batch([[1, 2], [3, 4]])

# Flat array for framework ingestion (padding/batching done downstream)
flat_ids, lengths = tok.encode_batch_to_array(["hello", "world"])
# flat_ids: np.ndarray[int32] — concatenated token IDs
# lengths: np.ndarray[int64] — number of tokens per input
```

### Streaming

Key differentiator. Streaming encode splits text at arbitrary byte boundaries
and produces tokens incrementally. Streaming decode handles partial UTF-8.

```python
# Encode stream — feed text chunks, get token IDs
with tok.encode_stream() as stream:
    ids: list[int] = stream.feed("Hello ")
    ids: list[int] = stream.feed("world!")
    ids: list[int] = stream.finalize()   # flush remaining

# Decode stream — feed token IDs, get text
with tok.decode_stream(skip_special_tokens=True) as stream:
    text: str = stream.feed([101, 7592])
    text: str = stream.feed([999])
    text: str = stream.finalize()

# Iterator for LLM inference (token-at-a-time decode)
for text_chunk in tok.decode_stream_iter(token_generator):
    print(text_chunk, end="", flush=True)
```

Streams own their internal state buffers and must be finalized. Context
managers handle cleanup. Streams can also be used manually:

```python
stream = tok.encode_stream()
ids = stream.feed(chunk)
ids = stream.finalize()
stream.close()  # or del stream
```

### Vocabulary

```python
tok.vocab_size: int              # number of tokens
tok.model_type: str              # "BPE" | "WordPiece" | "Unigram"
tok.token_to_id("hello") -> int | None
tok.id_to_token(42) -> str | None

# Special token IDs (None if not configured)
tok.bos_token_id: int | None
tok.eos_token_id: int | None
tok.unk_token_id: int | None
tok.pad_token_id: int | None
tok.sep_token_id: int | None
tok.cls_token_id: int | None
tok.mask_token_id: int | None
```

---

## API Comparison

| Operation | iree.tokenizer | HF tokenizers | tiktoken |
|-----------|---------------|---------------|----------|
| Load file | `Tokenizer.from_file(p)` | `Tokenizer.from_file(p)` | — |
| Load string | `Tokenizer.from_str(s)` | `Tokenizer.from_str(s)` | — |
| Encode | `tok.encode(text)` → `list[int]` | `tok.encode(text).ids` → `list[int]` | `enc.encode(text)` → `list[int]` |
| Encode numpy | `tok.encode_to_array(text)` | — | `enc.encode_to_numpy(text)` |
| Decode | `tok.decode(ids)` | `tok.decode(ids)` | `enc.decode(ids)` |
| Batch encode | `tok.encode_batch(texts)` | `tok.encode_batch(texts)` | `enc.encode_batch(texts)` |
| Batch decode | `tok.decode_batch(ids_list)` | `tok.decode_batch(ids_list)` | `enc.decode_batch(ids_list)` |
| Streaming encode | `tok.encode_stream()` | — | — |
| Streaming decode | `tok.decode_stream()` | `DecodeStream` (partial) | — |
| Vocab size | `tok.vocab_size` | `tok.get_vocab_size()` | `enc.n_vocab` |
| Token → ID | `tok.token_to_id(t)` | `tok.token_to_id(t)` | — |
| ID → Token | `tok.id_to_token(id)` | `tok.id_to_token(id)` | — |
| Special tokens | `tok.add_special_tokens=True` | `add_special_tokens=True` | `allowed_special=` |

---

## Architecture

### nanobind Binding Layer

```
src/bindings/
├── module.cc        NB_MODULE(_iree_tokenizer, m) entry point
├── tokenizer.cc     Tokenizer class: load, encode, decode, batch, vocab
├── streaming.cc     EncodeStream / DecodeStream classes
└── encoding.cc      Encoding result class (rich output)
```

### Ownership Model

**Tokenizer**: wraps `iree_tokenizer_t*` via `std::unique_ptr` with custom
deleter (`iree_tokenizer_free`). Immutable after construction. Shared across
threads without locking.

**Streams**: own heap-allocated `state_storage` and `transform_buffer` (encode
only). Hold a `nb::ref<Tokenizer>` preventing GC of parent tokenizer. Freed on
`close()` / context manager exit / destructor.

**Encode output**: `encode()` allocates a temporary C array, converts to Python
list. `encode_to_array()` allocates a numpy array, passes its `.data()` pointer
directly to the C API (zero-copy into numpy).

**Decode output**: pre-allocates buffer at `4 * token_count` bytes (heuristic),
retries with 2x on `IREE_STATUS_RESOURCE_EXHAUSTED`.

### Error Mapping

| C Status | Python Exception |
|----------|-----------------|
| `IREE_STATUS_INVALID_ARGUMENT` | `ValueError` |
| `IREE_STATUS_UNIMPLEMENTED` | `NotImplementedError` |
| `IREE_STATUS_NOT_FOUND` | `KeyError` |
| `IREE_STATUS_RESOURCE_EXHAUSTED` | Internal retry (encode/decode), `MemoryError` if unrecoverable |
| Other | `RuntimeError` with status message |

### C API Targets

The nanobind module links against these IREE CMake targets:
- `iree::tokenizer::tokenizer` — core encode/decode/batch/streaming
- `iree::tokenizer::format::huggingface::tokenizer_json` — JSON parser
- Transitive deps pulled in automatically (normalizers, segmenters, models, decoders, vocab)

---

## Build System

### Directory Layout

```
py-tokenizer/
├── CMakeLists.txt           Top-level CMake
├── pyproject.toml           PEP 517 config (scikit-build-core)
├── version.json             Version pins
├── LICENSE                  Apache-2.0 WITH LLVM-exception
├── src/
│   ├── iree/               Namespace package (no __init__.py)
│   │   └── tokenizer/
│   │       ├── __init__.py  Public API re-exports
│   │       ├── _core.pyi   Type stubs for native module
│   │       └── py.typed     PEP 561 marker
│   └── bindings/
│       ├── CMakeLists.txt   nanobind target
│       ├── module.cc
│       ├── tokenizer.cc
│       ├── streaming.cc
│       └── encoding.cc
├── tests/
│   ├── conftest.py          Fixtures (tokenizer.json loading)
│   ├── test_load.py
│   ├── test_encode.py
│   ├── test_decode.py
│   ├── test_batch.py
│   ├── test_streaming.py
│   ├── test_vocab.py
│   ├── test_array.py
│   └── test_consistency.py  Compare against HF tokenizers
├── tests/testdata/          tokenizer.json fixtures
├── benchmarks/
│   ├── bench_encode.py
│   ├── bench_decode.py
│   ├── bench_batch.py
│   ├── bench_streaming.py
│   └── bench_comparison.py  Head-to-head vs HF/tiktoken
├── docs/
│   └── design.md            This file
├── .github/workflows/
│   └── ci.yml
└── build_tools/
    └── sanitizers/
        ├── lsan_suppressions.txt
        └── ubsan_suppressions.txt
```

### CMakeLists.txt (top-level)

- CMake 3.24+, C17, C++20
- Reads `version.json` for package version and IREE pin
- IREE source: `-DIREE_SOURCE_DIR=<path>` or FetchContent from GitHub tag
- IREE flags: `BUILD_COMPILER=OFF`, `BUILD_TESTS=OFF`, `HAL_DRIVER_DEFAULTS=OFF`
- nanobind v2.9.0 via FetchContent
- `nanobind_add_module(_iree_tokenizer NB_STATIC LTO ...)`
- RTTI + exceptions enabled for binding sources (`-frtti -fexceptions`)
- ASAN/UBSAN options (Clang-only, following fusilli pattern)
- Install native module into `iree/tokenizer/` package directory

### pyproject.toml

```toml
[build-system]
requires = ["scikit-build-core>=0.10", "nanobind>=2.9.0"]
build-backend = "scikit_build_core.build"

[project]
name = "iree-tokenizer"
dynamic = ["version"]
description = "High-performance tokenizer for IREE"
license = "Apache-2.0 WITH LLVM-exception"
requires-python = ">=3.10"
dependencies = ["numpy>=1.26"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "Programming Language :: C++",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.optional-dependencies]
test = ["pytest>=7", "tokenizers", "tiktoken"]
bench = ["tokenizers", "tiktoken", "rich"]

[tool.scikit-build]
cmake.build-type = "Release"
wheel.packages = ["src/iree"]

[tool.scikit-build.cmake.define]
IREE_SOURCE_DIR = {env = "IREE_SOURCE_DIR", default = ""}

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### version.json

```json
{
  "package-version": "0.1.0.dev0",
  "iree-version": "3.11.0rc20260304"
}
```

---

## CI Workflow

Linux x86_64 only for v1.

### Matrix

| Job | Python | Build Type | Notes |
|-----|--------|-----------|-------|
| test-release | 3.10, 3.12 | Release | pytest suite |
| test-asan | 3.12 | Debug | ASAN+UBSAN, Clang |
| benchmark | 3.12 | Release | Record numbers, no gate |

### Steps

1. Checkout with submodules (IREE)
2. Install Python + build deps
3. Build wheel: `pip wheel . -w dist/`
4. Install wheel + test deps
5. Run `pytest tests/ -v`
6. (benchmark job) Run `python benchmarks/bench_comparison.py`
7. Upload benchmark results as artifact

---

## Benchmarks

### What to Measure

| Benchmark | Description |
|-----------|-------------|
| `bench_encode.py` | Single-string encode: short (10 words), medium (paragraph), long (10K chars). English + CJK + mixed. |
| `bench_decode.py` | Single-string decode: 100, 1K, 10K tokens. |
| `bench_batch.py` | Batch encode: 1, 10, 100, 1000 items. Throughput (tokens/sec). |
| `bench_streaming.py` | Streaming encode with 64-byte chunks vs 4K chunks vs one-shot. |
| `bench_comparison.py` | Same inputs across iree.tokenizer, HF tokenizers, tiktoken. |

### Models

- **GPT-2** (BPE + ByteLevel) — tiktoken `gpt2` / HF `openai-community/gpt2`
- **LLaMA-3** (BPE) — representative modern LLM tokenizer
- **BERT** (WordPiece) — `google-bert/bert-base-uncased`

### Metrics

- Tokens/sec (encode throughput)
- Latency p50, p99 (single encode)
- Memory: peak RSS via `tracemalloc`

### Output

Table printed to stdout with `rich`. JSON artifact saved for CI.

---

## Test Plan

| Test File | Coverage |
|-----------|----------|
| `test_load.py` | `from_file`, `from_str`, `from_buffer`, invalid JSON, missing file |
| `test_encode.py` | Basic encode, empty string, unicode (CJK, emoji, RTL), long text, `add_special_tokens` |
| `test_decode.py` | Round-trip (encode then decode), `skip_special_tokens`, empty list, single token |
| `test_batch.py` | `encode_batch`, `decode_batch`, empty batch, single-item, mixed lengths |
| `test_streaming.py` | `encode_stream`: feed chunks + finalize, byte-boundary splits, context manager, reuse after close error |
| | `decode_stream`: feed tokens + finalize, `decode_stream_iter` |
| `test_vocab.py` | `vocab_size`, `token_to_id`, `id_to_token`, unknown token, special IDs (bos/eos/unk/pad) |
| `test_array.py` | `encode_to_array` dtype=int32, `encode_batch_to_array` shapes, numpy interop |
| `test_consistency.py` | Compare encode output against HF tokenizers for GPT-2, BERT, LLaMA-3 fixtures |

### Test Fixtures

Ship `tokenizer.json` files in `tests/testdata/` for:
- GPT-2 (BPE + ByteLevel)
- BERT base uncased (WordPiece)
- A small synthetic tokenizer for fast unit tests

---

## Key Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Binding framework | nanobind | Required by project; matches IREE runtime bindings; lighter than pybind11 |
| Build backend | scikit-build-core | Modern PEP 517; first-class nanobind support |
| Default encode return | `list[int]` | Matches HF `.ids` and tiktoken convention |
| Batch impl | C batch API (shared state) | Single allocation, cache-friendly; no thread pool complexity |
| Streaming | Sync context managers | Clean lifecycle; async deferred to v2 |
| Namespace package | `iree.tokenizer` | Consistent with `iree.runtime`, `iree.compiler` |
| Token ID type | int32 numpy / int Python | Matches C API `int32_t`; Python int for list returns |
| No padding/truncation | Deferred | Framework responsibility (PyTorch DataLoader, JAX) |
| No `from_pretrained` | Deferred | Avoids runtime network deps; use `huggingface_hub` |
| No training | Out of scope | C library is inference-only |
| C++ standard | C++20 | Match fusilli; needed for nanobind features |
