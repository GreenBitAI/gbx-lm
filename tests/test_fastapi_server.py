import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import mlx.core as mx
import json
from argparse import Namespace
from gbx_lm.fastapi_server import create_app


class DummyTokenizer:
    def __init__(self):
        self.encode = MagicMock(side_effect=lambda text, *args, **kwargs: mx.array([1, 2, 3, 4, 5]))
        self.decode = MagicMock(return_value="Test response")
        self.eos_token_id = 50256
        self.detokenizer = MagicMock()
        self.detokenizer.text = "Test response"
        self.detokenizer.last_segment = "Test"
        self.detokenizer.reset = MagicMock()
        self.detokenizer.add_token = MagicMock()
        self.detokenizer.finalize = MagicMock()

        self.chat_template = "You are a helpful assistant.\n\nHuman: {input}\n\nAssistant:"
        self.apply_chat_template = MagicMock(side_effect=self._mock_apply_chat_template)

    def _mock_apply_chat_template(self, messages, *args, **kwargs):
        formatted_messages = self.chat_template.format(input=messages[-1]['content'])
        return self.encode(formatted_messages)


@patch('gbx_lm.fastapi_server.load')
def create_test_app(mock_load):
    # Create mock model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = DummyTokenizer()
    mock_load.return_value = (mock_model, mock_tokenizer)

    # Create test arguments
    args = Namespace(
        host="127.0.0.1",
        port=8000,
        model=None,
        model_list=["default_model", "chat_model"],
        adapter_path=None,
        trust_remote_code=False,
        chat_template="",
        use_default_chat_template=False,
        eos_token="<|eot_id|>",
        ue_parameter_path="test_params.db"
    )

    # Create app with test configuration
    app, _, _ = create_app(args)
    return app


class TestFastAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_test_app()
        cls.client = TestClient(cls.app)

    @patch('gbx_lm.fastapi_server.generate_step')
    @patch('gbx_lm.fastapi_server.async_generate_step')
    def test_create_completion(self, mock_async_generate_step, mock_generate_step):
        # Mock the async generator
        async def mock_generator(*args, **kwargs):
            yield ((mx.array([1]), mx.array([0.5]), None))

        mock_async_generate_step.return_value = mock_generator()
        mock_generate_step.return_value = [((mx.array([1]), mx.array([0.5]), None), 0)]  # 保留这个用于流式测试

        response = self.client.post(
            "/v1/default_model/completions",
            json={
                "model": "default_model",
                "prompt": "Once upon a time",
                "max_tokens": 10,
                "temperature": 0.5,
                "top_p": 0.9,
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("choices", data)
        self.assertEqual(data["object"], "text_completion")

    @patch('gbx_lm.fastapi_server.generate_step')
    @patch('gbx_lm.fastapi_server.async_generate_step')
    def test_create_chat_completion(self, mock_async_generate_step, mock_generate_step):
        # Mock the async generator
        async def mock_generator(*args, **kwargs):
            yield ((mx.array([1]), mx.array([0.5]), None))

        mock_async_generate_step.return_value = mock_generator()
        mock_generate_step.return_value = [((mx.array([1]), mx.array([0.5]), None), 0)]  # 保留这个用于流式测试

        response = self.client.post(
            "/v1/chat_model/chat/completions",
            json={
                "model": "chat_model",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "max_tokens": 10,
                "temperature": 0.7,
                "top_p": 0.85,
            }
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("choices", data)
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"]["role"], "assistant")
        self.assertIsNotNone(data["choices"][0]["message"]["content"])

    @patch('gbx_lm.fastapi_server.generate_step')
    def test_stream_completion(self, mock_generate_step):
        mock_generate_step.return_value = [((mx.array([1]), mx.array([0.5]), None), 0)]

        with self.client.stream(
                "POST",
                "/v1/default_model/completions",
                json={
                    "model": "default_model",
                    "prompt": "Once upon a time",
                    "max_tokens": 10,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "stream": True,
                }
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))

            events = list(response.iter_lines())
            self.assertTrue(len(events) > 0)
            non_empty_events = [event for event in events if event]
            for event in non_empty_events[:-1]:
                if event.startswith('data: '):
                    data = json.loads(event[6:])
                    self.assertIn("id", data)
                    self.assertIn("choices", data)
                    self.assertEqual(data["object"], "text_completion")

            self.assertEqual(non_empty_events[-1], 'data: [DONE]')

    @patch('gbx_lm.fastapi_server.generate_step')
    def test_stream_chat_completion(self, mock_generate_step):
        mock_generate_step.return_value = [((mx.array([1]), mx.array([0.5]), None), 0)]

        with self.client.stream(
                "POST",
                "/v1/chat_model/chat/completions",
                json={
                    "model": "chat_model",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"},
                    ],
                    "max_tokens": 10,
                    "temperature": 0.7,
                    "top_p": 0.85,
                    "stream": True,
                }
        ) as response:
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers['content-type'].startswith('text/event-stream'))

            events = list(response.iter_lines())
            self.assertTrue(len(events) > 0)
            non_empty_events = [event for event in events if event]
            for event in non_empty_events[:-1]:
                if event.startswith('data: '):
                    data = json.loads(event[6:])  # Remove 'data: ' prefix
                    self.assertIn("id", data)
                    self.assertIn("choices", data)
                    self.assertEqual(data["object"], "chat.completion.chunk")

            self.assertEqual(non_empty_events[-1], 'data: [DONE]')


if __name__ == "__main__":
    unittest.main()