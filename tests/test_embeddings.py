import json
import unittest
from unittest.mock import patch

from app.core.config import EmbeddingModelConfig
from app.core.errors import ModelProviderError
from app.services.ai.embeddings_runner import (
    OpenAICompatibleEmbeddingClient,
    RemoteEmbeddingSimilarityScorer,
    build_embedding_client,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class EmbeddingTests(unittest.TestCase):
    def test_builds_openai_compatible_embedding_request(self):
        config = EmbeddingModelConfig(
            enabled=True,
            provider="openai_compatible",
            model_name="embedding-model",
            base_url="https://embedding.example/v1",
            api_key="test-key",
            dimensions=256,
            max_input_chars=5,
        )
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "data": [
                        {"index": 1, "embedding": [0, 1, 0]},
                        {"index": 0, "embedding": [1, 0, 0]},
                    ],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                }
            )

        with patch("app.services.ai.embeddings_runner.urlopen", fake_urlopen):
            result = OpenAICompatibleEmbeddingClient(config).embed_texts(["abcdef", "xyz"])

        self.assertEqual(captured["url"], "https://embedding.example/v1/embeddings")
        self.assertEqual(captured["timeout"], 10.0)
        self.assertEqual(captured["payload"]["model"], "embedding-model")
        self.assertEqual(captured["payload"]["input"], ["abcde", "xyz"])
        self.assertEqual(captured["payload"]["dimensions"], 256)
        self.assertEqual(result.vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.assertEqual(result.usage.total_tokens, 7)

    def test_remote_similarity_scorer_returns_cosine_score(self):
        class FakeClient:
            def embed_texts(self, texts):
                self.texts = texts
                return type("Result", (), {"vectors": [[1.0, 0.0], [0.5, 0.0]]})()

        scorer = RemoteEmbeddingSimilarityScorer(FakeClient())

        self.assertEqual(scorer.score("query", "text"), 1.0)

    def test_disabled_embedding_uses_noop_client(self):
        client = build_embedding_client(EmbeddingModelConfig(enabled=False))

        self.assertEqual(client.embed_texts(["hello"]).vectors, [])

    def test_rejects_unknown_embedding_provider(self):
        with self.assertRaises(ModelProviderError):
            build_embedding_client(EmbeddingModelConfig(enabled=True, provider="unknown"))


if __name__ == "__main__":
    unittest.main()
