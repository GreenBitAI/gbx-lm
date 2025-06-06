# Copyright © 2024 Apple Inc.
# Additional code from GreenBitAI is licensed under the Apache 2.0 License.

"""
Adapted from a PyTorch implementation by David Grangier
Integrated version supporting both GreenBitAI and mlx-community models
"""

import argparse
import json
import logging
import os
import inspect
from importlib.metadata import version
from pathlib import Path
from typing import Optional, Dict, Any, List

import lm_eval
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from .utils import common_prefix_len


def _rstrip_until(s, untils):
    """Limit a string <s> to the first occurrence of any substring in untils."""
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]


def _pad_inputs(inputs):
    lengths = np.array([len(x) for x in inputs])
    maxlen = lengths.max()
    padded = np.stack(
        [np.pad(x, (0, maxlen - len(x))) for x in inputs],
        axis=0,
    )
    return mx.array(padded), mx.array(lengths)


def detect_model_type(path_or_hf_repo: str) -> str:
    """
    Detect the type of model to determine which loading and processing flow to use

    Args:
        path_or_hf_repo: Model path or HF repository name

    Returns:
        str: 'gbx' refers to the GreenBitAI model, 'mlx' refers to the mlx-community model
    """
    # Identify model types based on naming conventions
    if "GreenBitAI" in path_or_hf_repo or "gbx" in path_or_hf_repo.lower():
        return "gbx"
    elif "mlx-community" in path_or_hf_repo:
        return "mlx"
    else:
        # If you are unsure, use a more general loading method by default
        logging.warning(f"Unable to determine the type of model {path_or_hf_repo}, using default load flow")
        return "mlx"


def is_qwen3_model(path_or_hf_repo: str) -> bool:
    """
    检测是否为支持 enable_thinking 参数的 Qwen3 模型

    Args:
        path_or_hf_repo: 模型路径或HF仓库名

    Returns:
        bool: 是否为 Qwen3 模型
    """
    return "qwen3-" in path_or_hf_repo.lower() or "qwen-3-" in path_or_hf_repo.lower()


@register_model("mlxlm")
class MLXLM(LM):
    tokenizer_name = lm_eval.models.huggingface.HFLM.tokenizer_name

    def __init__(
            self,
            path_or_hf_repo: str,
            batch_size: int = 16,
            max_tokens: Optional[int] = None,
            use_chat_template: Optional[bool] = None,
            enable_thinking: Optional[bool] = None,  # 添加 enable_thinking 参数
    ) -> None:
        super().__init__()
        self._batch_size = batch_size

        self.model_type = detect_model_type(path_or_hf_repo)
        self.is_qwen3 = is_qwen3_model(path_or_hf_repo)
        self.enable_thinking = enable_thinking  # 存储 enable_thinking 参数

        print(f"Detected model type: {self.model_type}")
        if self.is_qwen3:
            print(f"Qwen3 model detected, enable_thinking={self.enable_thinking}")

        if self.model_type == "gbx":
            from .utils import load
        elif self.model_type == "mlx":
            from mlx_lm.utils import load

        self._model, self.tokenizer = load(path_or_hf_repo)
        self._max_tokens = max_tokens or self.tokenizer.model_max_length
        self.use_chat_template = use_chat_template
        if use_chat_template is None:
            self.use_chat_template = self.tokenizer.chat_template is not None

    def apply_chat_template(self, chat_history: List[Dict[str, str]], add_generation_prompt=True) -> str:
        """
        重写父类的 apply_chat_template 方法，增加对 enable_thinking 参数的支持

        Args:
            chat_history: 聊天历史消息列表
            add_generation_prompt: 是否添加生成提示

        Returns:
            str: 应用了模板的文本
        """
        # 检查是否为 Qwen3 模型并且支持 enable_thinking
        if self.is_qwen3 and hasattr(self.tokenizer, 'apply_chat_template'):
            result = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=self.enable_thinking
            )
            return result

        # 对于不支持 enable_thinking 的情况，使用默认方法
        result = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        return result

    def _score_fn(self, inputs, step_size: int = 64):
        if self.model_type == "gbx":
            from .models.base import create_causal_mask
            from .models.cache import make_prompt_cache
        elif self.model_type == "mlx":
            from mlx_lm.models.base import create_causal_mask
            from mlx_lm.models.cache import make_prompt_cache

        inputs, lengths = _pad_inputs(inputs)
        inputs, targets = inputs[..., :-1], inputs[..., 1:]

        cache = make_prompt_cache(self._model)

        scores, is_greedy = [], []
        for i in range(0, inputs.shape[1], step_size):
            inp = inputs[:, i: i + step_size]
            T = inp.shape[1]

            offset = cache[0].offset
            mask = create_causal_mask(T, offset, lengths=lengths)
            if self.is_qwen3 and self.model_type == "gbx":
                print("GBX Qwen3, converting mask to bf16 dtype.")
                mask = mask.astype(mx.bfloat16)
            logits = self._model(inp, cache=cache, mask=mask)
            log_probs = nn.log_softmax(logits.astype(mx.float32))

            score = mx.take_along_axis(
                log_probs, targets[:, i: i + step_size, mx.newaxis], axis=-1
            )[..., 0]
            ig = targets[:, i: i + step_size] == mx.argmax(logits, axis=-1)
            ig = mx.where(mx.arange(T) + offset < lengths[:, None], ig, False)

            mx.eval(score, ig)
            mx.clear_cache()

            is_greedy.append(ig)
            scores.append(score)

        scores = mx.concatenate(scores, axis=1)
        is_greedy = mx.concatenate(is_greedy, axis=1)

        return scores, lengths, is_greedy

    def _loglikelihood(self, texts, score_spans=None):
        all_scores = mx.zeros(len(texts))
        all_is_greedy = mx.zeros(len(texts), dtype=mx.bool_)
        for i in tqdm(range(0, len(texts), self._batch_size)):
            batch = texts[i: i + self._batch_size]
            scores, lengths, is_greedy = self._score_fn(batch)

            ind = np.arange(scores.shape[-1])
            if score_spans is not None:
                spans = score_spans[i: i + self._batch_size]
                lengths = [end - start for start, end in spans]
                masks = mx.array(
                    np.array([(ind >= start) & (ind < end) for start, end in spans])
                )
            else:
                masks = ind[None] < lengths[:, None]

            scores = (masks * scores).sum(axis=-1)
            is_greedy = (masks * is_greedy).sum(axis=-1)

            all_scores[i: i + self._batch_size] = scores
            all_is_greedy[i: i + self._batch_size] = is_greedy == lengths

        return all_scores, all_is_greedy

    def _tokenize(self, texts):
        """
        根据是否使用聊天模板对文本进行标记化
        """
        # 判断是否有需要特殊处理的JSON格式聊天消息
        processed_texts = []
        for text in texts:
            processed_texts.append(text)

        return [
            tuple(
                self.tokenizer.encode(t, add_special_tokens=not self.use_chat_template)
            )
            for t in processed_texts
        ]

    def loglikelihood(self, requests) -> List[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))

        # tokenize prefix and prefix + completion for all requests.
        tokenized = self._tokenize(
            [t for r in requests for t in [r.args[0], r.args[0] + r.args[1]]]
        )

        # max length (prefix + completion) and longest common prefix per question.
        length_stats = {}
        for prefix, completed in zip(tokenized[0::2], tokenized[1::2]):
            max_completed_l, min_prefix_l = length_stats.get(prefix, (0, 1e8))
            length_stats[prefix] = (
                max(max_completed_l, len(completed)),
                min(min_prefix_l, common_prefix_len(prefix, completed)),
            )

        # truncate requests for completed sequences longer than model context.
        shortened = []
        completion_spans = []
        long_completions = 0
        for prefix, completed in zip(tokenized[0::2], tokenized[1::2]):
            max_completed_l, prefix_l = length_stats[prefix]
            # compute truncation length
            truncation = max(0, max_completed_l - self._max_tokens - 1)
            prefix_l = prefix_l - truncation
            if prefix_l <= 0:
                # completion too long, prefix is eliminated for some requests.
                long_completions += 1
                truncation = max(0, len(completed) - self._max_tokens - 1)
                prefix_l = 1
            # truncate the completed sequence
            completed = completed[truncation:]
            shortened.append(completed)
            # scores do not include initial bos, substract 1 to span bounds
            completion_spans.append((prefix_l - 1, len(completed) - 1))

        if long_completions > 0:
            logging.info(
                f"Prefix eliminated for {long_completions} requests with "
                + "completion longer than context."
            )

        num_results = len(shortened)

        # sort by length to get batches with little padding.
        sorted_indices = sorted(range(len(shortened)), key=lambda i: -len(shortened[i]))
        shortened = [shortened[i] for i in sorted_indices]
        completion_spans = [completion_spans[i] for i in sorted_indices]

        group = mx.distributed.init()

        # split strided so we have approximately the same lengths on each node
        shortened = shortened[group.rank():: group.size()]
        completion_spans = completion_spans[group.rank():: group.size()]

        # model scoring, returns num_requests x (logp, is_greedy, length).
        scores, is_greedy = self._loglikelihood(
            shortened,
            score_spans=completion_spans,
        )

        # all gather the results across groups
        if group.size() > 1:
            per_group = int(np.ceil(num_results / group.size()))
            scores = mx.pad(scores, ((0, per_group - len(scores)),))
            is_greedy = mx.pad(is_greedy, ((0, per_group - len(is_greedy))))
            scores = mx.distributed.all_gather(scores[mx.newaxis], stream=mx.cpu)
            is_greedy = mx.distributed.all_gather(is_greedy[mx.newaxis], stream=mx.cpu)
            mx.eval(scores, is_greedy)
            scores = scores.T.reshape(-1)
            is_greedy = is_greedy.T.reshape(-1)

        scores = np.array(scores[:num_results])
        is_greedy = np.array(is_greedy[:num_results])

        results = [(score, ig) for score, ig in zip(scores, is_greedy)]
        inv_sort = np.argsort(sorted_indices)
        results = [results[inv_sort[i]] for i in range(len(inv_sort))]
        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:
                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3
                INPUT:    3   4   5   6
                PRED:     4   5   6   7
                INPUT:    5   6   7   8
                PRED:             8   9
          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the EOT token.
        """
        logging.info(
            "Estimating loglikelihood rolling for %d sequences." % len(requests)
        )
        inputs = self._tokenize([req.args[0] for req in requests])
        scores, _ = self._loglikelihood(inputs)
        return scores.tolist()

    def _process_thinking_output(self, text: str) -> str:
        """
        处理带有思考内容的输出，提取出非思考部分

        Args:
            text: 生成的文本

        Returns:
            str: 处理后的文本
        """
        if self.enable_thinking:
            # 保留思考内容，打印完整输出
            return text
        else:
            # 移除思考内容
            try:
                end_think_index = text.rfind('</think>')
                if end_think_index >= 0:
                    thinking_content = text[:end_think_index].strip()
                    content = text[end_think_index + len('</think>'):].strip()
                    logging.info(f"Thinking content detected and processed.")
                    return content
            except Exception as e:
                logging.error(f"Error processing thinking content: {e}")

        return text

    def generate_until(self, requests) -> List[str]:
        """Generate greedily until a stopping sequence
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        if self.model_type == "gbx":
            from .utils import stream_generate
        elif self.model_type == "mlx":
            from mlx_lm.generate import stream_generate

        logging.info("Generating continuation for %d sequences." % len(requests))
        contexts, options = zip(*[req.args for req in requests])
        # contrary to the doc the second element of the tuple contains
        # {'do_sample': False, 'until': ['\n\n'], 'temperature': 0}
        completions = []

        for context, opt in tqdm(zip(contexts, options), total=len(contexts)):
            until = opt["until"]

            # 检查是否为 JSON 格式聊天消息
            if self.use_chat_template and self.is_qwen3:
                try:
                    # 尝试解析为 JSON 格式的聊天消息
                    is_json_chat = False
                    messages = None

                    if context.startswith('[') and context.endswith(']'):
                        messages = json.loads(context)
                        is_json_chat = True
                    elif context.startswith('{') and context.endswith('}'):
                        message = json.loads(context)
                        if isinstance(message, dict) and 'role' in message and 'content' in message:
                            messages = [message]
                            is_json_chat = True

                    if is_json_chat and messages is not None:
                        # 使用我们的 apply_chat_template 方法，该方法会处理 enable_thinking
                        context = self.apply_chat_template(messages, add_generation_prompt=True)
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    # 不是 JSON 格式聊天消息，继续使用原始方式
                    pass

            context_tokens = self.tokenizer.encode(
                context, add_special_tokens=not self.use_chat_template
            )
            max_tokens = min(
                opt.get("max_gen_tokens", self._max_tokens),
                self.tokenizer.model_max_length - len(context_tokens),
            )
            text = ""
            for response in stream_generate(
                    self._model, self.tokenizer, prompt=context_tokens, max_tokens=max_tokens
            ):
                text += response.text
                if any(u in text for u in until):
                    text = _rstrip_until(text, until)
                    # 处理思考内容
                    if self.is_qwen3 and '</think>' in text:
                        text = self._process_thinking_output(text)
                    completions.append(text)
                    break
            else:
                # 处理思考内容
                if self.is_qwen3 and '</think>' in text:
                    text = self._process_thinking_output(text)
                completions.append(text)

        return completions


def main():
    parser = argparse.ArgumentParser(
        "Evaluate an MLX model using lm-evaluation-harness."
    )
    parser.add_argument("--model", help="Model to evaluate", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for result files."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-shots", type=int, default=0, help="Number of shots")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum nunber of tokens to generate. Defaults to the model's max context length.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Limit the number of examples per task.",
        type=int,
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Whether to provide the fewshot examples as a multiturn "
             "conversation or a single user turn.",
        default=False,
    )
    parser.add_argument(
        "--apply-chat-template",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to apply a chat template to the prompt. If "
             "the model has a chat template, this defaults to `True`, "
             "otherwise `False`.",
        default=None,
    )
    # 添加 enable_thinking 参数
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        help="Specifies whether to enable the thinking mode for Qwen3 models. "
             "When enabled, thinking content will be included in the output. "
             "When disabled, thinking content will be removed.",
        default=None,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mx.random.seed(args.seed)

    lm = MLXLM(
        args.model,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        use_chat_template=args.apply_chat_template,
        enable_thinking=args.enable_thinking,  # 传递 enable_thinking 参数
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        apply_chat_template=lm.use_chat_template,
        num_fewshot=args.num_shots,
        limit=args.limit,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    model_name = args.model.replace("/", "_")
    task_names = "_".join(args.tasks)
    ver = version("lm_eval")
    filename = f"eval_{model_name}_{task_names}_{args.num_shots:02d}_v_{ver}.json"
    if mx.distributed.init().rank() == 0:
        output_path = output_dir / filename
        output_path.write_text(json.dumps(results["results"], indent=4))
        print("Results:")
        for result in results["results"].values():
            print(json.dumps(result, indent=4))