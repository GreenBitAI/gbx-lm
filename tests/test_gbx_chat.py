import unittest
from unittest.mock import patch, MagicMock, call
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult, Generation
from gbx_lm.langchain import GBXPipeline, ChatGBX

class TestChatGBX(unittest.TestCase):
    def setUp(self):
        self.mock_llm = MagicMock(spec=GBXPipeline)
        self.mock_llm.tokenizer = MagicMock()
        self.mock_llm.pipeline_kwargs = {
            "temp": 0.7,
            "max_tokens": 100,
            "top_p": 1.0
        }
        self.mock_llm.model = MagicMock()
        self.mock_llm.max_kv_size = None
        self.mock_llm.cache_history = None
        self.chat_model = ChatGBX(llm=self.mock_llm)

        # Only mock _to_chat_result
        self.chat_model._to_chat_result = MagicMock(return_value=ChatResult(generations=[]))

    def test_initialization(self):
        self.assertIsInstance(self.chat_model, BaseChatModel)
        self.assertEqual(self.chat_model.tokenizer, self.mock_llm.tokenizer)

    def test_to_chatml_format(self):
        system_message = SystemMessage(content="System message")
        human_message = HumanMessage(content="Human message")
        ai_message = AIMessage(content="AI message")

        self.assertEqual(self.chat_model._to_chatml_format(system_message),
                         {"role": "system", "content": "System message"})
        self.assertEqual(self.chat_model._to_chatml_format(human_message),
                         {"role": "user", "content": "Human message"})
        self.assertEqual(self.chat_model._to_chatml_format(ai_message),
                         {"role": "assistant", "content": "AI message"})

        with self.assertRaises(ValueError):
            self.chat_model._to_chatml_format(MagicMock())

    def test_to_chat_prompt(self):
        messages = [
            SystemMessage(content="System message"),
            HumanMessage(content="Human message")
        ]
        self.chat_model.tokenizer.apply_chat_template.return_value = "Formatted prompt"

        result = self.chat_model._to_chat_prompt(messages)

        self.assertEqual(result, "Formatted prompt")
        self.chat_model.tokenizer.apply_chat_template.assert_called_once()

    def test_to_chat_prompt_validation(self):
        with self.assertRaises(ValueError):
            self.chat_model._to_chat_prompt([])

        with self.assertRaises(ValueError):
            self.chat_model._to_chat_prompt([SystemMessage(content="System message")])

    def test_llm_type(self):
        self.assertEqual(self.chat_model._llm_type, "gbx-chat-wrapper")

    def test_generate(self):
        messages = [HumanMessage(content="Hello")]

        # Create a proper LLMResult
        llm_result = LLMResult(
            generations=[[Generation(text="Hi there!")]],
            llm_output={"token_usage": {"total_tokens": 10}}
        )
        self.mock_llm._generate.return_value = llm_result

        result = self.chat_model._generate(messages)

        # Assertions
        self.mock_llm._generate.assert_called_once()
        self.chat_model._to_chat_result.assert_called_once_with(llm_result)

        self.assertIsInstance(result, ChatResult)

    def test_generate_with_stop(self):
        messages = [HumanMessage(content="Tell me a story")]
        stop = ["Once upon a time"]

        # Create a proper LLMResult
        llm_result = LLMResult(
            generations=[[Generation(text="Here's a story: Once upon a time...")]],
            llm_output={"token_usage": {"total_tokens": 15}}
        )
        self.mock_llm._generate.return_value = llm_result

        result = self.chat_model._generate(messages, stop=stop)

        # Assertions
        self.mock_llm._generate.assert_called_once()
        self.assertEqual(self.mock_llm._generate.call_args[1]['stop'], stop)
        self.chat_model._to_chat_result.assert_called_once_with(llm_result)

        self.assertIsInstance(result, ChatResult)

    @patch('gbx_lm.generate_step')
    @patch('mlx.core')
    def test_stream(self, mock_mx, mock_generate_step):
        messages = [HumanMessage(content="Hello")]
        mock_generate_step.return_value = iter([(1, 0.5, None), (2, 0.3, None), (3, 0.2, None)])

        self.chat_model.tokenizer.encode.return_value = [[1, 2, 3]]
        self.chat_model.tokenizer.eos_token_id = 3
        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "Test "
        self.chat_model.tokenizer.detokenizer = mock_detokenizer

        # Mock mx.array
        mock_mx.array.return_value = MagicMock()

        chunks = list(self.chat_model._stream(messages))

        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertIsInstance(chunk.message, AIMessageChunk)
            self.assertEqual(chunk.message.content, "Test ")

if __name__ == '__main__':
    unittest.main()