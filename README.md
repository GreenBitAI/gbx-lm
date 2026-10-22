# gbx-lm

> **Note:** the source code in this repository is an older version and is no
> longer where gbx-lm is developed. gbx-lm now ships as a signed binary, and
> the binary is the reference: use the [latest release](https://github.com/GreenBitAI/gbx-lm/releases/latest)
> rather than installing from this source.

gbx-lm is a local inference server for Apple Silicon Macs and, from v0.8.1,
the NVIDIA DGX Spark. It runs GreenBitAI's models -- including builds that page
their experts from disk and ship their own draft heads for speculative
decoding -- as well as MLX models
from [mlx-community](https://huggingface.co/mlx-community), and serves them
behind the APIs that existing tools already speak: OpenAI's Chat Completions
and Responses, and Anthropic's Messages.

## What's new in v0.8.2

- **Ternary Bonsai 2 27B, with a draft head of its own.** Prism ML's ternary
  27B runs on the Mac and on the DGX Spark, with a multi-token prediction head
  GreenBit AI trained for it on its own output
  ([GreenBitAI/Ternary-Bonsai-2-27B-gbx](https://huggingface.co/GreenBitAI/Ternary-Bonsai-2-27B-gbx)).
  Every drafted token is checked by the model, so the replies are the model's own.

  | Ternary Bonsai 2 27B, decode tok/s (short prompts) | gbx-lm + draft head | gbx-lm | llama.cpp (PQ2_0) | Prism ML's MLX runtime |
  | --- | --- | --- | --- | --- |
  | Mac Studio, M3 Ultra | **114** (136 on code) | 67 | 44 | 42 |
  | DGX Spark | **62** (80 on code) | 33 | 29.5 | -- |

  On the Mac that is 2.7x Prism ML's own runtime and 2.6x llama.cpp, and 1.7x
  the fastest other speculative decoder for it (MTPLX, 66 tok/s). Same machine,
  same prompts, greedy, for every runtime; details on the model card.
- **Bonsai 2 on the DGX Spark in 5.9 GB.** The ternary weights are held in a
  1.6-bit form after loading (12.7 GB before; llama.cpp's PQ2_0 file is 7.2 GB),
  and long prompts are read 1.6x as fast as llama.cpp's CUDA build (a 4k
  document at 1603 vs 1001 tok/s, 16k at 1523 vs 969).
- **Faster on the DGX Spark** (GB10, 128 GB). Prompt processing is up to 2.3x
  as fast as v0.8.1, and draft heads now stay ahead of plain decoding at long
  contexts (GLM-5.3 at 32k: 40 -> 53 tok/s with its head):

  | model | 4k prompt, tok/s | 16k prompt, tok/s | decode tok/s, plain / with draft head |
  | --- | --- | --- | --- |
  | Ternary-Bonsai-2-27B-gbx | **1603** (llama.cpp 1001) | **1523** (llama.cpp 969) | 33 / **62** |
  | Qwen3.8-27B-4bit | 669 -> **1175** | 637 -> **1167** | 13 / **25** |
  | Qwen3.6-35B-A3B-4bit-paged | 2617 -> **3362** | 2386 -> **3293** | 68 / **110** |
  | Qwen3.6-35B-A3B-8bit-paged | 2096 -> **2633** | 1968 -> **2609** | 49 / **74** |
  | Qwen3.8-Flash-Next-4bit-paged | 813 -> **966** | 364 -> **852** | 24 / **40** |

  Prompt columns: v0.8.1 -> v0.8.2. Decode: short prompts, averaged over code,
  prose and chat (code alone is faster with a draft head). All five run in
  memory on a DGX Spark.
- Fixes: a client leaving mid-stream no longer holds the GPU; Codex's
  `parallel_tool_calls: false` is accepted.

Earlier versions: see the [releases](https://github.com/GreenBitAI/gbx-lm/releases).

## Requirements

**Mac**

| | |
| --- | --- |
| macOS | **15.0 or later** |
| chip | **Apple Silicon** (arm64). There is no Intel build. |
| Python | none -- the binary carries what it needs |

**Linux** (from v0.8.1)

| | |
| --- | --- |
| machine | **NVIDIA DGX Spark** (GB10). Other Linux machines are not supported yet. |
| system | arm64, glibc 2.39 or later (DGX OS / Ubuntu 24.04) |
| driver | an NVIDIA driver with CUDA 13 support. No CUDA toolkit is needed: the binary carries the CUDA libraries it uses. |
| Python | none |

## Install

```bash
# upgrading? clear the previous version's unpack directory first
rm -rf ~/.libra/cache/onefile/gbx_lm

# on Linux (DGX Spark), use gbx_lm-linux-arm64.tar.gz in place of gbx_lm-darwin-arm64.tar.gz
curl -fL -o gbx_lm-darwin-arm64.tar.gz 'https://github.com/GreenBitAI/gbx-lm/releases/latest/download/gbx_lm-darwin-arm64.tar.gz' \
  && tar -xzf gbx_lm-darwin-arm64.tar.gz gbx_lm \
  && mkdir -p "$HOME/.local/bin" \
  && mv gbx_lm "$HOME/.local/bin/gbx_lm" \
  && chmod +x "$HOME/.local/bin/gbx_lm"

gbx_lm -h
```

The Mac build is signed with a Developer ID and notarised, so macOS runs it
without the usual detour for a downloaded binary.

**`command not found`** -- `$HOME/.local/bin` is not on your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc     # zsh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bash_profile && source ~/.bash_profile   # bash
```

**`Killed: 9`** (Mac) -- a previous version's files are still in the unpack
directory, and macOS refuses to mix two builds. Run the `rm -rf` line above, then try again.

## Run

```bash
gbx_lm --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged
```

That serves the model on port **11688**, the default; `--port` picks another.
The weights download on first use into `~/.libra/cache/models`. Set `HF_HOME` to
put them elsewhere, and `HF_TOKEN` if you meet the Hub's rate limits for
anonymous downloads.

```bash
curl http://127.0.0.1:11688/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GreenBitAI/Qwen3.8-Flash-Next-4bit-paged","messages":[{"role":"user","content":"Hello"}]}'
```

The server listens on every network interface (`0.0.0.0`) unless told
otherwise, so other machines on your network can reach it. To keep it to this
machine, add `--host 127.0.0.1`.

`gbx_lm -h` lists every option. The ones most often wanted:

| option | what it does |
| --- | --- |
| `--model` | the model to serve: a Hugging Face repo id or a local path |
| `--model_list` | several models at once, chosen per request by the `model` field |
| `--host` / `--port` | where to listen (default `0.0.0.0:11688`) |
| `--max_kv_size` | cap on the context kept in memory; sized from the machine's RAM if not given |
| `--prefill_step_size` | tokens per prefill step; lower uses less memory; sized from RAM if not given |
| `--log-level` | `TRACE` to `CRITICAL` (default `INFO`) |

## Command line

From `gbx_lm` v0.7.1, the binary also runs a model without a server, as a
subcommand -- one prompt, or an interactive chat:

```bash
GBX_QWEN4_MTP=on gbx_lm generate --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged --prompt "Hello" --max-tokens 2048
GBX_QWEN4_MTP=on gbx_lm chat --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged --max-tokens 2048
```

Keep `--max-tokens` generous: the default is 100 for `generate` and 256 for
`chat`, and these models reason before they answer, which counts against it.
`gbx_lm generate -h` and `gbx_lm chat -h` list the other options, and each model's
draft-head switch below works here as it does for the server. Anything else on
the command line starts the server; an older binary takes `generate` for a
server option and stops, so upgrade first.

## Models

### GreenBitAI builds

Each has a model card with its measurements and the details that are particular
to it. All seven carry a draft head in `mtp/`; it is off unless its switch is set.

| model | draft head switch | context |
| --- | --- | --- |
| [DeepSeek-V4.1-Flash-4bit-paged](https://huggingface.co/GreenBitAI/DeepSeek-V4.1-Flash-4bit-paged) | `GBX_DEEPSEEK_MTP=on` | 1,048,576 |
| [GLM-5.3-Flash-4bit-paged](https://huggingface.co/GreenBitAI/GLM-5.3-Flash-4bit-paged) | `GBX_GLM53_MTP=on` | 1,048,576 |
| [Qwen3.8-Flash-Next-4bit-paged](https://huggingface.co/GreenBitAI/Qwen3.8-Flash-Next-4bit-paged) | `GBX_QWEN4_MTP=on` | 262,144 |
| [Qwen3.6-35B-A3B-4bit-paged](https://huggingface.co/GreenBitAI/Qwen3.6-35B-A3B-4bit-paged) | `GBX_QWEN35_MTP=on` | 262,144 |
| [Qwen3.6-35B-A3B-8bit-paged](https://huggingface.co/GreenBitAI/Qwen3.6-35B-A3B-8bit-paged) | `GBX_QWEN35_MTP=on` | 262,144 |
| [Qwen3.8-27B-4bit](https://huggingface.co/GreenBitAI/Qwen3.8-27B-4bit) | `GBX_QWEN35_MTP=on` | 262,144 |
| [Ternary-Bonsai-2-27B-gbx](https://huggingface.co/GreenBitAI/Ternary-Bonsai-2-27B-gbx) | `GBX_QWEN35_MTP=on` | 262,144 |

They are collected at
[GreenBitAI for Apple Silicon and DGX Spark](https://huggingface.co/collections/GreenBitAI/greenbitai-for-apple-silicon-and-dgx-spark).

```bash
GBX_QWEN4_MTP=on gbx_lm --model GreenBitAI/Qwen3.8-Flash-Next-4bit-paged
```

Every token the draft head proposes is checked by the model itself, so the reply
is the model's own either way; the head only saves passes over the weights.

**Paging.** The `-paged` builds keep their routed experts in `experts.bin`.
Where the weights fit in memory they are loaded from it and the model runs at
full speed; where they do not, experts stream from disk as tokens need them --
slower, but the model runs. Reading the machine decides that, not a flag;
`GBX_PAGING=off` holds the experts resident regardless. On a DGX Spark,
GLM-5.3-Flash and DeepSeek-V4.1-Flash page; the other five run in memory.

### mlx-community models

Text models from [mlx-community](https://huggingface.co/mlx-community) whose
architecture [mlx-lm](https://github.com/ml-explore/mlx-lm) implements load as
they are:

```bash
gbx_lm --model mlx-community/Qwen3-0.6B-4bit
```

Checked with `mlx-community/Qwen3-0.6B-4bit` and
`mlx-community/Llama-3.2-1B-Instruct-4bit`. Checkpoints converted for other MLX
tools -- mlx-vlm, speech, embeddings -- are not supported, and a model stops on
the end-of-sequence tokens its own configuration declares.

## APIs

One port, three wire protocols:

| path | for |
| --- | --- |
| `/v1/chat/completions` | anything written against the OpenAI API |
| `/v1/responses` | **Codex** |
| `/v1/messages` | **Claude Code** |

### Thinking

For models that think before they answer, a Chat Completions request can switch
it and set its depth:

| field | example | notes |
| --- | --- | --- |
| `enable_thinking` | `true` / `false` | off when not given; GLM-5.3-Flash thinks either way, and takes only its depth |
| `thinking` | `{"type": "enabled"}` | the same switch, spelled as GLM spells it; `enable_thinking` wins if both are given |
| `reasoning_effort` | `"low"`, `"high"`, ... | a word the model's chat template rejects is refused with a 400 naming the ones it takes; a template with no depth setting ignores the field. DeepSeek-V4.1 also takes an integer from 1 to 100 |

The reasoning comes back in `reasoning_content`, apart from the answer.

`GBX_ENABLE_THINKING` and `GBX_REASONING_EFFORT` set the same two for every
request from the server side: a bare value (`GBX_REASONING_EFFORT=low`) is a
default the request can override, and a `force:` prefix
(`GBX_REASONING_EFFORT=force:low`) overrides the request.

### Codex

A provider in `~/.codex/config.toml`:

```toml
[model_providers.gbx]
name = "gbx-lm"
base_url = "http://127.0.0.1:11688/v1"
wire_api = "responses"
```

and a profile in `~/.codex/gbx.config.toml`:

```toml
model_provider = "gbx"
model = "GreenBitAI/Qwen3.8-Flash-Next-4bit-paged"
model_context_window = 262144
```

### Claude Code

`~/.claude/gbx.settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:11688",
    "ANTHROPIC_AUTH_TOKEN": "local",
    "ANTHROPIC_MODEL": "GreenBitAI/Qwen3.8-Flash-Next-4bit-paged",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GreenBitAI/Qwen3.8-Flash-Next-4bit-paged"
  }
}
```

Both clients ask for a small model for their own background work, so every name
in the settings has to be one this server is serving.

## Where things are kept

| path | holds |
| --- | --- |
| `~/.local/bin/gbx_lm` | the binary |
| `~/.libra/cache/onefile/gbx_lm` | the binary's unpacked runtime, recreated on the next start |
| `~/.libra/cache/models` | downloaded models, unless `HF_HOME` says otherwise |
| `~/.libra/cache/prompts` | saved prompt caches |

To uninstall, remove the binary and the unpack directory:

```bash
rm -f ~/.local/bin/gbx_lm
rm -rf ~/.libra/cache/onefile/gbx_lm
```

Models and prompt caches stay until you remove them as well.
