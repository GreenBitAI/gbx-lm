import unittest
from unittest.mock import patch, MagicMock
from gbx_lm.langchain import GBXPipeline
from langchain_core.outputs import GenerationChunk


class TestGBXPipeline(unittest.TestCase):


    @patch('gbx_lm.langchain.gbx_pipeline.generate')
    def test_call(self, mock_generate):
        # Mock the generate function to return a simple string
        mock_generate.return_value = "Test response"

        # Create a pipeline instance
        pipeline = GBXPipeline(model=MagicMock(), tokenizer=MagicMock())

        # Test the _call method
        response = pipeline._call("Test prompt")

        self.assertEqual(response, "Test response")
        mock_generate.assert_called_once()

    @patch('gbx_lm.generate_step')
    def test_stream(self, mock_generate_step):
        # Mock the generate_step function
        mock_generate_step.return_value = iter([(1, 0.5), (2, 0.3), (3, 0.2)])

        # Create a pipeline instance with mocked tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [[1, 2, 3]]
        mock_tokenizer.eos_token_id = 3
        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "Test "
        mock_tokenizer.detokenizer = mock_detokenizer

        pipeline = GBXPipeline(model=MagicMock(), tokenizer=mock_tokenizer)

        # Test the _stream method
        chunks = list(pipeline._stream("Test prompt"))

        # We expect 3 chunks because the method yields for each token,
        # including the eos_token
        self.assertEqual(len(chunks), 3)

        # Check that all chunks are of type GenerationChunk
        for chunk in chunks:
            self.assertIsInstance(chunk, GenerationChunk)

        # Check that the last chunk contains the eos_token
        self.assertEqual(chunks[-1].text, "Test ")  # Assuming the detokenizer returns "Test " for the eos_token

        # Verify that generate_step was called with correct arguments
        mock_generate_step.assert_called_once()
        args, kwargs = mock_generate_step.call_args
        self.assertIn('prompt', kwargs)
        self.assertIn('model', kwargs)
        self.assertEqual(kwargs.get('top_p', None), 1.0)

    def test_identifying_params(self):
        pipeline = GBXPipeline(model_id="test_model", model=MagicMock(), tokenizer=MagicMock())
        params = pipeline._identifying_params

        self.assertIn('model_id', params)
        self.assertEqual(params['model_id'], "test_model")

    def test_llm_type(self):
        pipeline = GBXPipeline(model=MagicMock(), tokenizer=MagicMock())
        self.assertEqual(pipeline._llm_type, "gbx_pipeline")


if __name__ == '__main__':
    unittest.main()