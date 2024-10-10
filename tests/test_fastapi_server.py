import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from gbx_lm.fastapi_server import app, ModelProvider
import mlx.core as mx
import json


class DummyModelProvider:
    def __init__(self):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.tokenizer.encode.side_effect = self.mock_encode
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.eos_token_id = 50256
        self.tokenizer.detokenizer = MagicMock()
        self.tokenizer.detokenizer.text = "Test response"
        self.tokenizer.detokenizer.last_segment = "Test"

        self.tokenizer.chat_template = "You are a helpful assistant.\n\nHuman: {input}\n\nAssistant:"
        self.tokenizer.apply_chat_template = self.mock_apply_chat_template

    def mock_encode(self, text, *args, **kwargs):
        return mx.array([1, 2, 3, 4, 5])

    def mock_apply_chat_template(self, messages, *args, **kwargs):
        formatted_messages = self.tokenizer.chat_template.format(input=messages[-1]['content'])
        return self.mock_encode(formatted_messages)

    def load(self, model, adapter=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


def mock_generate_step(*args, **kwargs):
    yield (mx.array([1]), mx.array([0.5])), None, None


class TestFastAPIServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_provider = DummyModelProvider()
        cls.client = TestClient(app)

    @patch('gbx_lm.fastapi_server.model_provider', new_callable=DummyModelProvider)
    @patch('gbx_lm.fastapi_server.generate_step', mock_generate_step)
    def test_create_completion(self, mock_model_provider):
        response = self.client.post(
            "/v1/completions",
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

    @patch('gbx_lm.fastapi_server.model_provider', new_callable=DummyModelProvider)
    @patch('gbx_lm.fastapi_server.generate_step', mock_generate_step)
    def test_create_chat_completion(self, mock_model_provider):
        response = self.client.post(
            "/v1/chat/completions",
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

    @patch('gbx_lm.fastapi_server.model_provider', new_callable=DummyModelProvider)
    @patch('gbx_lm.fastapi_server.generate_step', mock_generate_step)
    def test_stream_completion(self, mock_model_provider):
        with self.client.stream(
                "POST",
                "/v1/completions",
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
                    data = json.loads(event[6:])  # Remove 'data: ' prefix
                    self.assertIn("id", data)
                    self.assertIn("choices", data)
                    self.assertEqual(data["object"], "text_completion")

            self.assertEqual(non_empty_events[-1], 'data: [DONE]')

    @patch('gbx_lm.fastapi_server.model_provider', new_callable=DummyModelProvider)
    @patch('gbx_lm.fastapi_server.generate_step', mock_generate_step)
    def test_stream_chat_completion(self, mock_model_provider):
        with self.client.stream(
                "POST",
                "/v1/chat/completions",
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