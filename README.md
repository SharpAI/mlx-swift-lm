# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

> [!IMPORTANT]
> The `main` branch is a _new_ major version number: 3.x.  In order
> to decouple from tokenizer and downloader packages some breaking
> changes were introduced. See [upgrading documentation](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/upgrade) for detailed instructions on upgrading.
>
> If that page shows a 404 you can view the source:
> [upgrading](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/upgrade.md) 
> and [using](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md)

Some key features include:

- Model loading with integrations for a variety of tokenizer and model downloading packages.
- Low-rank (LoRA) and full model fine-tuning with support for quantized models.
- Many model architectures for both LLMs and VLMs.

For some example applications and tools that use MLX Swift LM, check out [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

## Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [Techniques for developing in mlx-swift-lm](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/developing)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon): Common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm): Large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm): Vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders): Popular encoders and embedding models example implementations

## Usage

This package integrates with a variety of tokenizer and downloader packages through protocol conformance. Users can pick from three ways to integrate with these packages, which offer different tradeoffs between freedom and convenience.

See documentation on [how to integrate mlx-swift-lm and downloaders/tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).

## Installation

Add the core package to your `Package.swift`:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

Then chose an [integration package for downloaders and tokenizers](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


## Quick Start

After installing the package you can use LLMs to generate content with only a few lines
of code.  (Note: the exact line to load the model depends on the [integration package](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages)).

> [!NOTE]
> If the documentation link shows a 404, view the
> [source](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Documentation.docc/using.md).


```swift
import MLXLLM
import MLXLMCommon

let modelConfiguration = LLMRegistry.gemma3_1B_qat_4bit

// customize this line per the integration package
let model = try await loadModelContainer(
    configuration: modelConfiguration
)

let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Using the [adapter packages](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/using#Integration-Packages) you would have similar code -- replace the imports and the load line.

For example, loading from a local directory using the [DePasqualeOrg/swift-tokenizers-mlx](https://github.com/DePasqualeOrg/swift-tokenizers-mlx):

```swift
import MLXLLM
import MLXLMTokenizers

let modelDirectory = URL(filePath: "/path/to/model")
let container = try await loadModelContainer(
    from: modelDirectory,
    using: TokenizersLoader()
)
```

## Performance

Benchmarks run on a **MacBook Pro M5 Pro, 64 GB unified memory** using the built-in automated profiler (`run_benchmark.sh → Test 1`).

### DeepSeek-V4-Flash (126 GB, Q3-mixed-gs128-affine)

Model: [`Thump604/DeepSeek-V4-Flash-MLX-Q3-mixed-gs128-affine`](https://huggingface.co/Thump604/DeepSeek-V4-Flash-MLX-Q3-mixed-gs128-affine)

> Dense/Vanilla and TurboQuant (non-SSD) configurations are skipped automatically — the 126 GB model exceeds available physical RAM and would cause system instability.

| Configuration | Context | TTFT | Speed | GPU Alloc (virtual) | GPU InUse peak (physical) |
|---|---|---|---|---|---|
| SSD Stream | 512 | 6.80 s | 4.65 tok/s | 28.4 GB | 16.7 GB |
| SSD Stream | 40,000 | 565 s | 0.32 tok/s | 60.5 GB | 12.5 GB |
| **SSD + TurboQuant** | **512** | **6.35 s** | **4.78 tok/s** | **29.5 GB** | **16.8 GB** |
| **SSD + TurboQuant** | **40,000** | **364 s** | **4.16 tok/s** | **40.6 GB** | **16.8 GB** |
| SSD + 16-Worker Prefetch | 512 | 5.84 s | 4.43 tok/s | 29.3 GB | 16.6 GB |
| SSD + 16-Worker Prefetch | 40,000 | 566 s | 0.32 tok/s | 60.9 GB | 13.6 GB |

**Key findings:**
- **SSD + TurboQuant is the clear winner** — 4.2 tok/s at 40K context vs 0.32 tok/s for baseline SSD Stream (13× faster), and 36% lower GPU virtual allocation (40.6 GB vs 60.5 GB).
- **GPU InUse (physical RAM)** is the peak physical RAM high-water mark, sampled every 0.5 s during prefill + generation. GPU Alloc is the total virtual GPU address space including SSD-backed pages — the true memory demand.
- At 512-token context all three SSD configurations perform similarly (~4.4–4.8 tok/s). TurboQuant's advantage emerges strongly at long context where KV cache compression matters most.

### Running the Benchmark Yourself

```bash
# Build the release binary first
swift build -c release

# Launch the interactive benchmark suite
./run_benchmark.sh
# → Select: 1) Test 1: Automated Context & Memory Profile
# → Select: 11) Thump604/DeepSeek-V4-Flash-MLX-Q3-mixed-gs128-affine
# → Context lengths: 512,40000
```

Results are saved to `docs/profiling/profiling_results_<hostname>.md`.
