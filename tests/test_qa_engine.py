import unittest
from pathlib import Path

from src.document_loader import load_folder
from src.qa_engine import DocumentQA, FALLBACK_MESSAGE

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample_corpus"


class DocumentQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qa = DocumentQA.from_segments(load_folder(SAMPLE_DIR))

    def test_supported_question_returns_answer_and_source(self):
        result = self.qa.answer("How long may an undergraduate borrow a book?")
        self.assertTrue(result["supported"])
        self.assertIn("14 days", result["answer"])
        self.assertTrue(any("library_policy.txt" in item for item in result["citations"]))
        self.assertTrue(result["passages"])

    def test_unsupported_question_uses_required_fallback(self):
        result = self.qa.answer("What is the parking fee at the college?")
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], FALLBACK_MESSAGE)
        self.assertEqual(result["citations"], [])

    def test_empty_question_uses_fallback(self):
        result = self.qa.answer("   ")
        self.assertFalse(result["supported"])
        self.assertEqual(result["answer"], FALLBACK_MESSAGE)


if __name__ == "__main__":
    unittest.main()
