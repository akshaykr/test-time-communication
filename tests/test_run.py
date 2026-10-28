import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run


class TokenSelectionTests(unittest.TestCase):
    def test_token_precedence_and_isolation(self):
        cases = [
            ({"GH_TOKEN": "test-gh-token"}, None, "test-gh-token"),
            ({"COPILOT_GITHUB_TOKEN": "", "GH_TOKEN": "test-gh-token"}, None, "test-gh-token"),
            ({"COPILOT_GITHUB_TOKEN": "test-copilot-token", "GH_TOKEN": "test-gh-token"},
             "test-file-token", "test-copilot-token"),
            ({"GH_TOKEN": "test-gh-token"}, "test-file-token", "test-gh-token"),
            ({"GH_TOKEN": ""}, "test-file-token", "test-file-token"),
            ({}, None, None),
        ]
        for variables, file_token, expected in cases:
            with self.subTest(variables=variables, file_token=file_token):
                with tempfile.TemporaryDirectory() as home:
                    if file_token is not None:
                        token_file = Path(home) / ".copilot/pat"
                        token_file.parent.mkdir()
                        token_file.write_text(file_token + "\n")
                    with patch.dict(
                        os.environ, {**variables, "GITHUB_TOKEN": "test-github-token"}, clear=True,
                    ), patch.object(run.Path, "home", return_value=Path(home)):
                        args = run.parser().parse_args([
                            "polyomino", "--model", "test-model", "--dry-run",
                        ])
                        cmd, env, _ = run.prepare(args)
                    self.assertEqual(env.get("COPILOT_GITHUB_TOKEN"), expected)
                    self.assertNotIn("GH_TOKEN", env)
                    self.assertNotIn("GITHUB_TOKEN", env)
                    if expected:
                        self.assertNotIn(expected, " ".join(cmd))


if __name__ == "__main__":
    unittest.main()
