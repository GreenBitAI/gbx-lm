"""
A model worker using Apple MLX

https://github.com/ml-explore/mlx-examples/tree/main/llms

Code based on vllm_worker https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/vllm_worker.py

You must install MLX python:

pip install mlx-lm
"""

import argparse
import asyncio
import atexit
import json
from typing import List, Optional
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)

import mlx.core as mx
from gbx_lm import load, generate_step
from gbx_lm.server_utils import (
    stopping_criteria,
    sequence_overlap
)
from gbx_lm.sample_utils import make_sampler

app = FastAPI()


class MLXWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: "MLX",
        conv_template: str,
        trust_remote_code: True or None = None,
        eos_token: str = None,
        adapter_file: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 256
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: MLX worker..."
        )

        self.model_name = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Building tokenizer_config
        tokenizer_config = {"trust_remote_code": True if trust_remote_code else None}
        if eos_token is not None:
            tokenizer_config["eos_token"] = eos_token

        self.mlx_model, self.mlx_tokenizer = load(
            model_path, adapter_path=adapter_file, tokenizer_config=tokenizer_config
        )

        self.tokenizer = self.mlx_tokenizer
        # self.context_len = get_context_length(
        #     llm_engine.engine.model_config.hf_config)
        self.context_len = 2048  # hard code for now -- not sure how to get in MLX

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", self.temperature))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = params.get("max_new_tokens", self.max_tokens)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        for tid in stop_token_ids:
            if tid is not None:
                s = self.tokenizer.decode(tid)
                if s != "":
                    stop.add(s)

        print("Stop patterns: ", stop)
        context_mlx = mx.array(self.tokenizer.encode(context))

        finish_reason = "length"

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        sampler = make_sampler(temperature, top_p)

        iterator = await run_in_threadpool(
            generate_step, context_mlx, self.mlx_model, max_tokens=max_new_tokens, sampler=sampler
        )

        stop_id_sequences = [self.tokenizer.encode(st) for st in stop]
        tokens = []
        for i in range(max_new_tokens):
            (token, prob, _) = await run_in_threadpool(next, iterator)

            if token in self.tokenizer.eos_token_ids:
                break
            stop_condition = stopping_criteria(tokens, stop_id_sequences, self.tokenizer.eos_token_id)
            if stop_condition.stop_met:
                finish_reason = "stop"
                break

            detokenizer.add_token(token)
            tokens.append(token)

            if any(sequence_overlap(tokens, sequence) for sequence in stop_id_sequences):
                continue

            ret = {
                "text": detokenizer.text,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": len(context),
                    "completion_tokens": len(detokenizer.tokens),
                    "total_tokens": len(context) + len(detokenizer.tokens),
                },
                "cumulative_logprob": [],
                "finish_reason": None,  # hard code for now
            }

            yield (json.dumps(ret) + "\0").encode()

        detokenizer.finalize()
        ret = {
            "text": detokenizer.text,
            "error_code": 0,
            "usage": {},
            "cumulative_logprob": [],
            "finish_reason": finish_reason,
        }
        yield (json.dumps(obj={**ret, **{"finish_reason": None}}) + "\0").encode()
        yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        print("trying to abort but not implemented")

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = uuid.uuid4()
    params["request_id"] = str(request_id)
    output = await worker.generate(params)
    release_worker_semaphore()
    # await engine.abort(request_id)
    print("Trying to abort but not implemented")
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


worker = None


def cleanup_at_exit():
    global worker
    print("Cleaning up...")
    del worker


atexit.register(cleanup_at_exit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    parser.add_argument("--model-path", type=str, default="microsoft/phi-2")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when"
        "downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="End of sequence token for tokenizer",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        help="Optional path for the trained adapter weights.",
    )
    parser.add_argument("--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for model output (default: 1024)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )

    args, unknown = parser.parse_known_args()

    if args.model_path:
        args.model = args.model_path

    worker = MLXWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        1024,
        False,
        "MLX",
        args.conv_template,
        trust_remote_code = args.trust_remote_code,
        eos_token = args.eos_token,
        adapter_file = args.adapter_file,
        temperature = args.temperature,
        max_tokens=args.max_tokens
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
