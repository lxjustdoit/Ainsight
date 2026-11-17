import unittest

from app.services.ai.utils.search_process_util import SearchFindingsConfig, SearchFindingsEvaluator, is_valid_page_title


class SearchFindingsTests(unittest.TestCase):
    def test_na_values_are_not_valid_findings(self):
        evaluator = SearchFindingsEvaluator(SearchFindingsConfig(min_chars=3))

        self.assertTrue(evaluator.is_na("N/A"))
        self.assertTrue(evaluator.is_na("n/a"))
        self.assertFalse(evaluator.is_valid(search_findings="N/A", question="AI summaries"))

    def test_validates_findings_with_local_similarity(self):
        evaluator = SearchFindingsEvaluator(SearchFindingsConfig(min_chars=20, similarity_threshold=0.1))

        self.assertTrue(
            evaluator.is_valid(
                search_findings=(
                    "1. Title: AI summaries\n"
                    "URL: https://example.com/ai\n"
                    "Summary: AI summaries help users understand dense pages."
                ),
                question="How do AI summaries help users?",
            )
        )

    def test_validates_multilingual_findings_with_char_ngram_similarity(self):
        evaluator = SearchFindingsEvaluator(SearchFindingsConfig(min_chars=10, similarity_threshold=0.08))

        self.assertTrue(
            evaluator.is_valid(
                search_findings="1. 标题：AI 阅读总结\n摘要：AI 总结工具可以帮助用户理解复杂网页内容。",
                question="AI总结工具如何帮助阅读网页？",
            )
        )

    def test_invalid_title_does_not_lower_question_similarity(self):
        evaluator = SearchFindingsEvaluator(SearchFindingsConfig(min_chars=10, similarity_threshold=0.08))

        self.assertFalse(is_valid_page_title("Home"))
        self.assertTrue(
            evaluator.is_valid(
                search_findings="AI summaries help users understand dense pages and reading workflows.",
                question="How do AI summaries help users read pages?",
                page_title="Home",
            )
        )

    def test_valid_title_can_improve_similarity(self):
        evaluator = SearchFindingsEvaluator(SearchFindingsConfig(min_chars=10, similarity_threshold=0.08))

        self.assertTrue(is_valid_page_title("Ainsight reading assistant"))
        self.assertTrue(
            evaluator.is_valid(
                search_findings="Ainsight reading assistant creates structured page summaries.",
                question="What does this tool do?",
                page_title="Ainsight reading assistant",
            )
        )

    def test_compress_limits_results_and_total_length(self):
        evaluator = SearchFindingsEvaluator(
            SearchFindingsConfig(max_results=2, max_chars_per_result=80, max_total_chars=170)
        )
        text = (
            "1. Title: AI summaries\n"
            "URL: https://example.com/ai\n"
            "Summary: AI summaries help users understand dense pages. Extra unrelated sentence about cooking.\n\n"
            "2. Title: Reading tools\n"
            "URL: https://example.com/reading\n"
            "Summary: Reading tools organize long articles. Extra unrelated sentence about weather.\n\n"
            "3. Title: Third result\n"
            "URL: https://example.com/third\n"
            "Summary: This should be clipped."
        )

        compressed = evaluator.compress(search_findings=text, question="AI summaries reading tools")

        self.assertLessEqual(len(compressed), 170)
        self.assertIn("AI summaries", compressed)
        self.assertNotIn("Third result", compressed)


if __name__ == "__main__":
    unittest.main()
