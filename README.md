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

## What's new in v0.8.3

- **Long conversations on the DGX Spark, about twice as fast.** A conversation
  that already holds a long document now answers each new message in half the
  time or less -- the shape of a coding agent's work. With a ~54k-token
  document in the conversation:

  | DGX Spark | first answer, v0.8.2 -> v0.8.3 | next 1.3k-token message, v0.8.2 -> v0.8.3 |
  | --- | --- | --- |
  | Ternary-Bonsai-2-27B-gbx | 92 s -> **42 s** | 4.5 s -> **1.6 s** |
  | Qwen3.8-27B-4bit | 101 s -> **53 s** | 4.5 s -> **2.3 s** |
  | Qwen3.6-35B-A3B-4bit-paged | 38 s -> **14 s** | 1.6 s -> **0.8 s** |
  | Qwen3.8-Flash-Next-4bit-paged | 117 s -> **54 s** | 4.9 s -> **2.2 s** |

- **Faster prompts and decoding on the DGX Spark.** Prompt processing on the
  Qwen3.6-35B builds is up to 1.8x as fast, and Qwen3.8-Flash-Next no longer
  slows down on long prompts:

  | DGX Spark | 4k prompt, tok/s | 16k prompt, tok/s | 54k prompt, tok/s | decode tok/s, plain / with draft head (code) |
  | --- | --- | --- | --- | --- |
  | Qwen3.6-35B-A3B-4bit-paged | 3845 -> **5694** | 3780 -> **5462** | 3262 -> **4409** | 64 -> **71** / 133 -> **141** |
  | Qwen3.6-35B-A3B-8bit-paged | 2838 -> **5015** | 2829 -> **4927** | 2531 -> **4059** | 48 -> **53** / 88 -> **93** |
  | Qwen3.8-Flash-Next-4bit-paged | 1028 -> **1274** | 911 -> **1243** | 480 -> **1065** | 23 -> 24 / 46 -> 46 |
  | Qwen3.8-27B-4bit | 1186 -> 1193 | 1173 -> 1180 | 1048 -> 1052 | 13 / 27 -> **34** |
  | Ternary-Bonsai-2-27B-gbx | 1620 -> 1623 | 1568 -> 1565 | 1261 -> 1338 | 33 / 76 -> 76 |

  v0.8.2 -> v0.8.3, one request at a time, greedy.
- **DFlash 2 for Qwen3.8-27B, as an option.** Its repository now also carries
  Inco AI's DFlash 2 draft, which proposes seven tokens in one pass:
  `GBX_QWEN35_MTP=on GBX_QWEN35_DRAFT=dflash`. On code it decodes faster than
  the model's own draft head -- 140 vs 114 tok/s on a Mac Studio (M3 Ultra), 45
  vs 34 on a DGX Spark -- and slower on prose, so the head stays the default.
- **On the Mac**, speeds are those of v0.8.2, and later messages in a long
  conversation come back about 10% sooner.
- Fixes: Ternary Bonsai 2 on the DGX Spark no longer pauses about 3 s the first
  time it meets each new prompt length (every turn of an agent did); on the DGX
  Spark the server's memory no longer grows with every turn of a long conversation
  (about 0.9 GiB per 7k-token turn before); a model in a
  local directory (`--model /path/to/model`) now loads, from the command line and
  in the server ([#20](https://github.com/GreenBitAI/gbx-lm/issues/20)).

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
| [DeepSeek-V4.1-Flash-1017-4bit-paged](https://huggingface.co/GreenBitAI/DeepSeek-V4.1-Flash-1017-4bit-paged) | `GBX_DEEPSEEK_MTP=on` | 1,048,576 |
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

**DFlash 2 for Qwen3.8-27B** (v0.8.3 or later). The 27B repository also carries
Inco AI's DFlash 2 draft in `dflash/` (1.3 GB, read only when asked for). It
proposes seven tokens in one pass of a small model: faster than the model's own
head on code and maths, slower on prose and chat, so it is a choice and not the
default. Text only; a request with images is served without drafting.

```bash
GBX_QWEN35_MTP=on GBX_QWEN35_DRAFT=dflash gbx_lm --model GreenBitAI/Qwen3.8-27B-4bit
```

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
