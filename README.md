# iree-tokenizer

Python bindings for the [IREE](https://github.com/iree-org/iree) tokenizer — a
high-performance C library supporting BPE, WordPiece, and Unigram models with
full HuggingFace `tokenizer.json` compatibility.

## Performance

All benchmarks: GPT-2 tokenizer, 200 iterations, p50 latency.
Machine: x86_64 Linux. Lower is better.

### Encode (22K chars → 5000 tokens)

<!--
iree: 438µs, tiktoken: 1220µs, hf: 5279µs
Normalized to max (hf = 100%)
iree: 8.3%, tiktoken: 23.1%, hf: 100%
-->

<p align="center">
<svg width="480" height="120" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; }
    .label { fill: #c9d1d9; font-weight: 500; }
    .value { fill: #8b949e; font-size: 12px; }
  </style>
  <text x="90" y="26" text-anchor="end" class="label">iree</text>
  <rect x="96" y="12" width="32" height="22" rx="3" fill="#58a6ff"/>
  <text x="136" y="28" class="value">438 µs  (12x faster)</text>
  <text x="90" y="62" text-anchor="end" class="label">tiktoken</text>
  <rect x="96" y="48" width="88" height="22" rx="3" fill="#f0883e"/>
  <text x="192" y="64" class="value">1,220 µs  (4.3x faster)</text>
  <text x="90" y="98" text-anchor="end" class="label">hf tokenizers</text>
  <rect x="96" y="84" width="380" height="22" rx="3" fill="#da3633"/>
  <text x="300" y="100" class="value" fill="#c9d1d9">5,279 µs</text>
</svg>
</p>

### Decode (5000 tokens → text)

<!--
iree: 28µs, tiktoken: 76µs, hf: 574µs
-->

<p align="center">
<svg width="480" height="120" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; }
    .label { fill: #c9d1d9; font-weight: 500; }
    .value { fill: #8b949e; font-size: 12px; }
  </style>
  <text x="90" y="26" text-anchor="end" class="label">iree</text>
  <rect x="96" y="12" width="19" height="22" rx="3" fill="#58a6ff"/>
  <text x="123" y="28" class="value">28 µs  (20x faster)</text>
  <text x="90" y="62" text-anchor="end" class="label">tiktoken</text>
  <rect x="96" y="48" width="50" height="22" rx="3" fill="#f0883e"/>
  <text x="154" y="64" class="value">76 µs  (7.6x faster)</text>
  <text x="90" y="98" text-anchor="end" class="label">hf tokenizers</text>
  <rect x="96" y="84" width="380" height="22" rx="3" fill="#da3633"/>
  <text x="300" y="100" class="value" fill="#c9d1d9">574 µs</text>
</svg>
</p>

### Batch Encode Throughput (100 items)

<!--
iree: 10.1M tok/s, tiktoken: 3.8M tok/s, hf: 0.87M tok/s
Normalized to iree = 100%
-->

<p align="center">
<svg width="480" height="120" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: system-ui, -apple-system, sans-serif; font-size: 13px; }
    .label { fill: #c9d1d9; font-weight: 500; }
    .value { fill: #8b949e; font-size: 12px; }
  </style>
  <text x="90" y="26" text-anchor="end" class="label">iree</text>
  <rect x="96" y="12" width="380" height="22" rx="3" fill="#58a6ff"/>
  <text x="300" y="28" class="value" fill="#c9d1d9">10.1M tokens/sec</text>
  <text x="90" y="62" text-anchor="end" class="label">tiktoken</text>
  <rect x="96" y="48" width="143" height="22" rx="3" fill="#f0883e"/>
  <text x="247" y="64" class="value">3.8M tokens/sec</text>
  <text x="90" y="98" text-anchor="end" class="label">hf tokenizers</text>
  <rect x="96" y="84" width="33" height="22" rx="3" fill="#da3633"/>
  <text x="137" y="100" class="value">0.87M tokens/sec</text>
</svg>
</p>

Run benchmarks yourself:
```bash
pip install tokenizers tiktoken rich huggingface-hub
python benchmarks/bench_comparison.py
```

## Install

```bash
# Build from source (requires CMake 3.24+, Ninja, C++20 compiler)
git clone --recurse-submodules https://github.com/iree-org/iree.git /path/to/iree
IREE_SOURCE_DIR=/path/to/iree pip install .
```

## Quick Start

```python
from iree.tokenizer import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

# Encode / decode
ids = tok.encode("Hello world")          # [15496, 995]
text = tok.decode(ids)                    # "Hello world"

# Batch
tok.encode_batch(["Hello", "world"])      # [[15496], [995]]

# Numpy arrays (zero-copy)
arr = tok.encode_to_array("Hello world")  # int32 ndarray

# Rich encoding with byte offsets
enc = tok.encode_rich("Hello world", track_offsets=True)
# enc.ids, enc.offsets, enc.type_ids

# Streaming decode (LLM inference pattern)
from iree.tokenizer import decode_stream_iter
for chunk in decode_stream_iter(tok, token_generator):
    print(chunk, end="", flush=True)
```

## API

| Method | Returns | Description |
|--------|---------|-------------|
| `Tokenizer.from_file(path)` | `Tokenizer` | Load from `tokenizer.json` |
| `Tokenizer.from_str(json)` | `Tokenizer` | Load from JSON string |
| `Tokenizer.from_buffer(bytes)` | `Tokenizer` | Load from bytes |
| `tok.encode(text)` | `list[int]` | Encode text to token IDs |
| `tok.encode_to_array(text)` | `np.ndarray` | Encode to numpy int32 array |
| `tok.encode_rich(text)` | `Encoding` | IDs + byte offsets + type IDs |
| `tok.decode(ids)` | `str` | Decode token IDs to text |
| `tok.encode_batch(texts)` | `list[list[int]]` | Batch encode |
| `tok.decode_batch(id_lists)` | `list[str]` | Batch decode |
| `tok.encode_stream()` | `EncodeStream` | Streaming encoder (context manager) |
| `tok.decode_stream()` | `DecodeStream` | Streaming decoder (context manager) |
| `tok.vocab_size` | `int` | Vocabulary size |
| `tok.model_type` | `str` | `"BPE"`, `"WordPiece"`, or `"Unigram"` |
| `tok.token_to_id(token)` | `int \| None` | Look up token ID |
| `tok.id_to_token(id)` | `str \| None` | Look up token text |

## Development

```bash
# Configure + build
cmake -B build -G Ninja -DIREE_SOURCE_DIR=/path/to/iree
cmake --build build

# Run tests (symlink .so into package)
ln -s build/_iree_tokenizer*.so src/iree/tokenizer/
PYTHONPATH=src pytest tests/ -v

# Run tests under ASAN (requires Clang)
cmake -B build-asan -G Ninja \
  -DIREE_SOURCE_DIR=/path/to/iree \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_TOKENIZER_ENABLE_ASAN=ON
cmake --build build-asan
ln -sf build-asan/_iree_tokenizer*.so src/iree/tokenizer/
LD_PRELOAD=$(clang++ -print-file-name=libclang_rt.asan.so) \
  ASAN_OPTIONS=detect_leaks=0 PYTHONPATH=src pytest tests/ -v
```

## License

Apache 2.0 with LLVM Exceptions — see [LICENSE](LICENSE).
