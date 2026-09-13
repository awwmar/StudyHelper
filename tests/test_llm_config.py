"""Model regression tests without provider requests or embedding downloads."""
import importlib
import os
import unittest
from unittest.mock import MagicMock, patch


class ModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"GROQ_API_KEY": "test-only"}, clear=True)
        self.env.start()
        self.dotenv = patch("dotenv.load_dotenv")
        self.dotenv.start()
        self.chat = patch("langchain_groq.ChatGroq")
        self.factory = self.chat.start()
        self.module = importlib.reload(importlib.import_module("services.llm"))
        self.factory.reset_mock()

    def tearDown(self):
        self.chat.stop()
        self.dotenv.stop()
        self.env.stop()

    def test_defaults_use_distinct_replacement_models(self):
        service = self.module.LLMService()
        self.assertEqual(service.model_name, "openai/gpt-oss-20b")
        self.assertEqual(service.fallback_model_name, "openai/gpt-oss-120b")

    def test_environment_overrides_and_matching_custom_model(self):
        with patch.dict(os.environ, {"GROQ_MODEL": "primary", "GROQ_FALLBACK_MODEL": "backup"}):
            service = self.module.LLMService()
        self.assertEqual([c.kwargs["model_name"] for c in self.factory.call_args_list], ["primary", "backup"])
        service.invoke_custom("hello", model_id="primary")
        self.assertEqual(self.factory.call_count, 2)

    def test_custom_prompt_falls_back_after_primary_failure(self):
        service = self.module.LLMService()
        service.llm = MagicMock()
        service.fallback_llm = MagicMock()
        service.llm.invoke.side_effect = RuntimeError("model unavailable")
        service.fallback_llm.invoke.return_value.content = "answer"
        self.assertEqual(service.invoke_custom("question"), "answer")
        service.fallback_llm.invoke.assert_called_once_with("question")

    def test_empty_model_rejected(self):
        with patch.dict(os.environ, {"GROQ_MODEL": " "}):
            with self.assertRaises(ValueError):
                self.module.LLMService()


if __name__ == "__main__":
    unittest.main()
