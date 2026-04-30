import os
import unittest


class TestRoadmappHtmlStructure(unittest.TestCase):
    def test_phase_tab_ids_not_duplicated(self):
        here = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(here, os.pardir))
        p = os.path.join(repo_root, "roadmapp.html")
        with open(p, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertEqual(html.count('id="tab-prog-0"'), 1)
        self.assertEqual(html.count('id="tab-prog-1"'), 1)
        self.assertEqual(html.count('id="tab-prog-2"'), 1)
        self.assertNotIn("switchPhase(3)", html)
