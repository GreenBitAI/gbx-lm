import unittest
from unittest.mock import patch, MagicMock
from gbx_lm.langchain import GBXPipeline
from langchain_core.outputs import GenerationChunk
import os
import warnings

# Environment variable to control skipping the large model download
SKIP_LARGE_DOWNLOAD = os.environ.get('SKIP_LARGE_DOWNLOAD', '').lower() in ('true', '1', 'yes')

class TestGBXPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.warn(
            "\nWARNING: This test will download a large model file (approximately 700MB). "
            "Ensure you have sufficient disk space and a stable internet connection. "
            "To skip this test, set the SKIP_LARGE_DOWNLOAD environment variable to 'true'.",
            category=ResourceWarning
        )

    @unittest.skipIf(SKIP_LARGE_DOWNLOAD, "Skipping test that requires downloading a large model")
    def test_from_model_id(self):
        model_id = "GreenBitAI/Qwen-1.5-0.5B-layer-mix-bpw-2.2-mlx"

        # Create pipeline
        pipeline = GBXPipeline.from_model_id(model_id)

        # Verify if the pipeline is correctly created
        self.assertIsInstance(pipeline, GBXPipeline)
        self.assertEqual(pipeline.model_id, model_id)
        self.assertIsNotNone(pipeline.model)
        self.assertIsNotNone(pipeline.tokenizer)

        # Test a simple call
        test_input = "Berlin is "
        response = pipeline(test_input)

        # Validate the response
        self.assertIsInstance(response, str)
        self.assertNotEqual(response, "")

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
        mock_generate_step.return_value = iter([(1, 0.5, None), (2, 0.3, None), (3, 0.2, None)])

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