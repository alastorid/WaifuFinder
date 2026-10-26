#!/usr/bin/env python3
"""Standard-library regression tests; fixtures are text, never executed software."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("license_audit", Path(__file__).with_name("audit-licenses.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
# Construct inert fixture strings without declaring dependencies in this test.
media_tool = "ff" + "mpeg"


class LicenseAuditTests(unittest.TestCase):
    def scan(self, text, suffix=".sh"):
        with tempfile.TemporaryDirectory(prefix="waifu-license-audit-") as directory:
            root = Path(directory)
            source = root / "tools" / ("fixture" + suffix)
            source.parent.mkdir()
            source.write_text(text)
            return audit.audit(root)[0]

    def test_embedded_python_denylist_does_not_declare_a_dependency(self):
        script = Path(__file__).with_name("setup-trufor-coreml-conversion.sh").read_text()
        self.assertEqual(self.scan(script), [])

    def test_membership_guard_masks_only_the_rejected_literals(self):
        for function in ["any", "all"]:
            for values in [f'("{media_tool}", "avcodec")', f'["{media_tool}", "avcodec"]']:
                self.assertEqual(self.scan(f'if {function}(token in name.lower() for token in {values}):\n    raise SystemExit("Rejected")'), [])

    def test_real_calls_remain_rejected(self):
        for line in [
            f'subprocess.run(["{media_tool}", "-version"])',
            f'Process.executableURL = URL(fileURLWithPath: "/usr/local/bin/{media_tool}")',
            f'brew install {media_tool}',
            f'import {media_tool}',
            f'{media_tool} -version',
        ]:
            self.assertTrue(self.scan(line), line)

    def test_a_call_inside_the_checked_expression_is_not_masked(self):
        line = f'if any(token in subprocess.run(["{media_tool}"]) for token in ("bad",)):\n    pass'
        self.assertTrue(self.scan(line))

    def test_arbitrary_lists_and_nonliteral_guards_still_fail(self):
        self.assertTrue(self.scan(f'commands = ["{media_tool}"]'))
        self.assertTrue(self.scan(f'if any(token in name for token in ("{media_tool}", lookup())):\n    pass'))

    def test_binary_and_package_license_checks_remain_active(self):
        with tempfile.TemporaryDirectory(prefix="waifu-license-audit-") as directory:
            root = Path(directory)
            (root / media_tool).write_bytes(b"fixture")
            metadata = root / "fixture.dist-info" / "METADATA"
            metadata.parent.mkdir()
            metadata.write_text("Name: fixture\nLicense-Expression: " + "LG" + "PL-3.0\n")
            errors = audit.audit(root)[0]
            self.assertTrue(any("Prohibited executable/library artifact" in item for item in errors))
            self.assertTrue(any("Prohibited package license" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
