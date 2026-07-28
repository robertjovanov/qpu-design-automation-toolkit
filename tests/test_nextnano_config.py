import importlib
import io
import os
import sys
import types
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout


class FakeConfig:
    def __init__(self):
        self.fullpath = "/unused/fake-nextnanopy-config"
        self.to_default_calls = 0
        self.set_calls = []
        self.save_calls = 0

    def to_default(self):
        self.to_default_calls += 1

    def set(self, section, key, value):
        self.set_calls.append((section, key, value))

    def save(self):
        self.save_calls += 1


@contextmanager
def import_with_fake_nextnanopy(fake_config):
    missing = object()
    previous_config_module = sys.modules.pop("nextnano_config", missing)
    previous_nextnanopy_module = sys.modules.get("nextnanopy", missing)

    fake_nextnanopy = types.ModuleType("nextnanopy")
    fake_nextnanopy.config = fake_config
    sys.modules["nextnanopy"] = fake_nextnanopy

    try:
        yield importlib.import_module("nextnano_config")
    finally:
        sys.modules.pop("nextnano_config", None)
        if previous_config_module is not missing:
            sys.modules["nextnano_config"] = previous_config_module

        if previous_nextnanopy_module is missing:
            sys.modules.pop("nextnanopy", None)
        else:
            sys.modules["nextnanopy"] = previous_nextnanopy_module


class NextnanoConfigTests(unittest.TestCase):
    def test_import_does_not_mutate_save_or_print(self):
        import_config = FakeConfig()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with redirect_stdout(stdout), redirect_stderr(stderr):
            with import_with_fake_nextnanopy(import_config):
                pass

        self.assertEqual(import_config.to_default_calls, 0)
        self.assertEqual(import_config.set_calls, [])
        self.assertEqual(import_config.save_calls, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_configure_nextnano_sets_expected_values_on_injected_config(self):
        import_config = FakeConfig()
        configured = FakeConfig()
        license_directory = "/test/licenses"
        output_directory = "/test/output"
        nextnano_directory = "/test/nextnano"

        with import_with_fake_nextnanopy(import_config) as module:
            module.configure_nextnano(
                license_directory=license_directory,
                output_directory=output_directory,
                nextnano_directory=nextnano_directory,
                save=False,
                config=configured,
            )

        self.assertEqual(configured.to_default_calls, 1)
        self.assertEqual(
            configured.set_calls,
            [
                ("nextnano++", "outputdirectory", output_directory),
                ("nextnano3", "outputdirectory", output_directory),
                ("nextnano.NEGF", "outputdirectory", output_directory),
                ("nextnano.MSB", "outputdirectory", output_directory),
                (
                    "nextnano3",
                    "license",
                    os.path.join(license_directory, "license.txt"),
                ),
                (
                    "nextnano++",
                    "license",
                    os.path.join(license_directory, "license.txt"),
                ),
                (
                    "nextnano.NEGF",
                    "license",
                    os.path.join(license_directory, "License_nnNEGF.lic"),
                ),
                (
                    "nextnano++",
                    "exe",
                    os.path.join(
                        nextnano_directory,
                        "nextnano++/bin/nextnano++_gcc_macOS_old",
                    ),
                ),
                (
                    "nextnano++",
                    "database",
                    os.path.join(
                        nextnano_directory,
                        "nextnano++/database/database.nnp",
                    ),
                ),
                (
                    "nextnano3",
                    "exe",
                    os.path.join(
                        nextnano_directory,
                        "nextnano3/bin/nextnano3_gcc_macOS_old",
                    ),
                ),
                (
                    "nextnano3",
                    "database",
                    os.path.join(
                        nextnano_directory,
                        "nextnano3/database/database.nn3",
                    ),
                ),
                (
                    "nextnano.NEGF",
                    "exe",
                    os.path.join(
                        nextnano_directory,
                        "nextnano.NEGF/bin/nextnano.NEGF_win.exe",
                    ),
                ),
                (
                    "nextnano.NEGF",
                    "database",
                    os.path.join(
                        nextnano_directory,
                        "nextnano.NEGF/database/Material_Database.in",
                    ),
                ),
                (
                    "nextnano.MSB",
                    "database",
                    os.path.join(
                        nextnano_directory,
                        "nextnano.MSB/database/materials.msb",
                    ),
                ),
            ],
        )

    def test_save_false_does_not_save(self):
        import_config = FakeConfig()
        configured = FakeConfig()

        with import_with_fake_nextnanopy(import_config) as module:
            module.configure_nextnano(config=configured, save=False)

        self.assertEqual(configured.save_calls, 0)

    def test_save_true_saves_exactly_once(self):
        import_config = FakeConfig()
        configured = FakeConfig()

        with import_with_fake_nextnanopy(import_config) as module:
            module.configure_nextnano(config=configured, save=True)

        self.assertEqual(configured.save_calls, 1)

    def test_main_applies_and_saves_configuration_explicitly(self):
        configured = FakeConfig()
        stdout = io.StringIO()

        with import_with_fake_nextnanopy(configured) as module:
            with redirect_stdout(stdout):
                module.main()

        self.assertEqual(configured.to_default_calls, 1)
        self.assertEqual(configured.save_calls, 1)
        self.assertIn(configured.fullpath, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
