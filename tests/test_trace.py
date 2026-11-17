import re
import unittest

from app.utils.id_generator import new_llm_call_id, new_trace_id


class TraceIdTests(unittest.TestCase):
    def test_trace_id_format_and_uniqueness(self):
        ids = {new_trace_id() for _ in range(200)}

        self.assertEqual(len(ids), 200)
        for trace_id in ids:
            self.assertRegex(trace_id, r"^[0-9a-f]{32}$")

    def test_llm_call_id_format(self):
        self.assertTrue(re.match(r"^llm_[0-9a-f]{32}$", new_llm_call_id()))


if __name__ == "__main__":
    unittest.main()
