"""gbx_ml Chat Wrapper."""

from typing import Any, List, Optional, Iterator
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser


from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)

from gbx_lm.langchain.gbx_pipeline import GBXPipeline

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class ChatGBX(BaseChatModel):
    """GBX chat models.

    Works with `GBXPipeline` LLM.

    To use, you should have the ``gbx-lm`` python package installed.

    Example:
        .. code-block:: python

            from gbx_lm.langchain import ChatGBX
            from gbx_lm.langchain import GBXPipeline

            llm = GBXPipeline.from_model_id(
                model_id="GreenBitAI/Llama-3-8B-layer-mix-bpw-4.0-mlx",
            )
            chat = ChatGBX(llm=llm)

    """

    llm: GBXPipeline
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokenizer = self.llm.tokenizer

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
        tokenize: bool = False,
        return_tensors: Optional[str] = None,
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts,
            tokenize=tokenize,
            add_generation_prompt=True,
            return_tensors=return_tensors,
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        # Ensure llm_output is a dictionary
        llm_output = llm_result.llm_output if isinstance(llm_result.llm_output, dict) else {}

        return ChatResult(
            generations=chat_generations, llm_output=llm_output
        )

    @property
    def _llm_type(self) -> str:
        return "gbx-chat-wrapper"

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            import mlx.core as mx
            from gbx_lm import generate_step

        except ImportError:
            raise ImportError(
                "Could not import gbx_lm python package. "
                "Please install it with `pip install gbx_lm`."
            )

        model_kwargs = kwargs.get("model_kwargs", self.llm.pipeline_kwargs) or {}
        temp: float = model_kwargs.get("temp", 0.7)
        max_new_tokens: int = model_kwargs.get("max_tokens", 100)
        repetition_penalty: Optional[float] = model_kwargs.get(
            "repetition_penalty", None
        )
        repetition_context_size: Optional[int] = model_kwargs.get(
            "repetition_context_size", None
        )
        top_p: float = model_kwargs.get("top_p", 1.0)

        llm_input = self._to_chat_prompt(messages, tokenize=True, return_tensors="np")

        prompt_tokens = mx.array(llm_input[0])

        eos_token_id = self.tokenizer.eos_token_id
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()

        for (token, prob, _), n in zip(
            generate_step(
                prompt_tokens,
                self.llm.model,
                temp,
                repetition_penalty,
                repetition_context_size,
                top_p=top_p,
                max_kv_size=self.llm.max_kv_size,
                cache_history=self.llm.cache_history
            ),
            range(max_new_tokens),
        ):
            # identify text to yield
            text: Optional[str] = None
            detokenizer.add_token(token)
            detokenizer.finalize()
            text = detokenizer.last_segment

            # yield text, if any
            if text:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

            # break if stop sequence found
            if token == eos_token_id or (stop is not None and text in stop):
                break

    def bind_tools(
            self, tools: List[Any], tool_choice: Optional[str] = None
    ) -> BaseChatModel:
        tool_strings = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                tool_strings.append(f"- {tool.name}: {tool.description}")
            elif isinstance(tool, type) and issubclass(tool, BaseModel):
                # This is a Pydantic model for structured output
                parser = PydanticOutputParser(pydantic_object=tool)
                tool_strings.append(f"Output Format: {parser.get_format_instructions()}")
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

        tool_string = "\n".join(tool_strings)

        new_system_message = SystemMessage(content=f"{self.system_message.content}\n\n{tool_string}")

        return self.__class__(
            llm=self.llm,
            system_message=new_system_message,
            tokenizer=self.tokenizer
        )