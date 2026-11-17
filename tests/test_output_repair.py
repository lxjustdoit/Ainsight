import unittest

from app.services.ai.output_repair import OutputRepairFailure, parse_repaired_output
from app.services.ai.pydantic_outputs import GraphOutput, QAOutput, QAPrepareOutput, SummaryOutput


class OutputRepairTests(unittest.TestCase):
    def test_parses_json_inside_markdown_fence(self):
        output = parse_repaired_output(
            '```json\n{"answer":"Current page only."}\n```',
            output_type=QAOutput,
            task_type="qa",
        )

        self.assertEqual(output.answer, "Current page only.")

    def test_wraps_plain_text_for_qa(self):
        output = parse_repaired_output("Current page only.", output_type=QAOutput, task_type="qa")

        self.assertEqual(output.answer, "Current page only.")

    def test_repairs_legacy_qa_prepare_final_answer_field(self):
        output = parse_repaired_output(
            '{"search_findings":"","final_answer":"Prepared answer.","feedback":""}',
            output_type=QAPrepareOutput,
            task_type="qa_prepare",
        )

        self.assertEqual(output.answer, "Prepared answer.")
        self.assertEqual(output.search_findings, "")

    def test_repairs_graph_mermaid_and_optional_tree_fields(self):
        output = parse_repaired_output(
            """
{
  "type": "mindmap",
  "mermaid": "```mermaid\\nroot((Topic))\\n```",
  "fallback_tree": {
    "label": "Topic",
    "children": [{"label": "Child"}]
  }
}
""",
            output_type=GraphOutput,
            task_type="graph",
        )

        self.assertTrue(output.mermaid.startswith("mindmap"))
        self.assertIsNone(output.fallback_tree.children[0].summary)
        self.assertEqual(output.fallback_tree.children[0].children, [])

    def test_summary_missing_core_fields_requires_regeneration(self):
        with self.assertRaises(OutputRepairFailure) as context:
            parse_repaired_output(
                '{"title":"Page","language":"en","overall_summary":"Summary."}',
                output_type=SummaryOutput,
                task_type="summary",
            )

        self.assertIn("sections", context.exception.feedback.missing_fields)


if __name__ == "__main__":
    unittest.main()
