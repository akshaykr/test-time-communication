import os
import shlex
import subprocess
import unittest
from unittest.mock import patch

from agent import CopilotAgent


class CopilotCommandTests(unittest.TestCase):
    def helpers(self):
        with patch.dict(os.environ, {"RELEASE_CONTINUATION": "Continue working."}):
            return CopilotAgent._copilot_action_feedback_helpers(None)

    def test_effort_flag_matches_cli_version(self):
        for option in ("--effort", "--reasoning-effort"):
            for long_context in ("0", "1"):
                with self.subTest(option=option, long_context=long_context):
                    help_text = f"  {option} <level>\n  --context <tier>\n"
                    script = self.helpers() + f"""
fake_copilot() {{ printf %s {shlex.quote(help_text)}; }}
copilot_binary=fake_copilot
RELEASE_LONG_CONTEXT={long_context}
configure_copilot_command
printf '%s\\n%s\\n' "$effort_option" "$context_flag"
"""
                    result = subprocess.run(
                        ["bash", "-c", script],
                        text=True, capture_output=True, check=True,
                    )
                    context = "--context long_context" if long_context == "1" else ""
                    self.assertEqual(result.stdout, f"{option}\n{context}\n")

    def test_older_cli_without_context_flag(self):
        script = self.helpers() + """
fake_copilot() { printf '%s\\n' '  --effort <level>'; }
copilot_binary=fake_copilot
RELEASE_LONG_CONTEXT=1
configure_copilot_command
printf '%s\\n' "$context_flag"
"""
        result = subprocess.run(
            ["bash", "-c", script], text=True, capture_output=True, check=True,
        )
        self.assertEqual(result.stdout, "\n")

    def test_help_failure_is_propagated(self):
        script = self.helpers() + """
fake_copilot() { echo 'help failed' >&2; return 7; }
copilot_binary=fake_copilot
configure_copilot_command
"""
        result = subprocess.run(
            ["bash", "-c", script], text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 7)
        self.assertIn("help failed", result.stderr)


if __name__ == "__main__":
    unittest.main()
