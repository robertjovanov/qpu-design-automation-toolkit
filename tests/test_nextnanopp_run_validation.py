import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import nextnanopp_tools as nnt


class RunDirectoryValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.temporary_root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_run(
        self,
        name="device_run",
        *,
        complete=True,
        biases=("bias_00000",),
    ):
        run_root = self.temporary_root / name
        run_root.mkdir()
        if complete:
            (run_root / "job_done.txt").write_text("done\n", encoding="utf-8")
        for bias_name in biases:
            (run_root / bias_name).mkdir()
        return run_root

    @staticmethod
    def make_output(parent, relative_path, text="data\n"):
        path = parent / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def snapshot_tree(root):
        paths = [root, *sorted(root.rglob("*"))]
        snapshot = []
        for path in paths:
            metadata = path.lstat()
            snapshot.append(
                (
                    str(path.relative_to(root)),
                    path.is_dir(),
                    metadata.st_mode,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    path.read_bytes() if path.is_file() else None,
                )
            )
        return snapshot

    def test_accepts_string_path(self):
        run_root = self.make_run()

        result = nnt.validate_run_directory(str(run_root))

        self.assertEqual(result, run_root.resolve())

    def test_accepts_path_object(self):
        run_root = self.make_run()

        result = nnt.validate_run_directory(run_root)

        self.assertEqual(result, run_root.resolve())

    def test_expands_user_home(self):
        run_root = self.make_run(name="home_run")

        with patch.dict(os.environ, {"HOME": str(self.temporary_root)}):
            result = nnt.validate_run_directory("~/home_run")

        self.assertEqual(result, run_root.resolve())

    def test_rejects_empty_string(self):
        for empty_value in ("", " \t "):
            with self.subTest(empty_value=empty_value):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    nnt.validate_run_directory(empty_value)

    def test_rejects_nonexistent_path(self):
        missing = self.temporary_root / "missing_run"

        with self.assertRaisesRegex(
            FileNotFoundError,
            str(missing.resolve()),
        ):
            nnt.validate_run_directory(missing)

    def test_rejects_file_instead_of_directory(self):
        ordinary_file = self.temporary_root / "not_a_run"
        ordinary_file.write_text("not a directory\n", encoding="utf-8")

        with self.assertRaisesRegex(
            NotADirectoryError,
            str(ordinary_file.resolve()),
        ):
            nnt.validate_run_directory(ordinary_file)

    def test_accepts_valid_completed_run_root(self):
        run_root = self.make_run()

        self.assertEqual(
            nnt.validate_run_directory(run_root),
            run_root.resolve(),
        )

    def test_returns_normalized_absolute_path_object(self):
        run_root = self.make_run()
        path_with_parent_segment = (
            run_root.parent / run_root.name / ".." / run_root.name
        )

        result = nnt.validate_run_directory(path_with_parent_segment)

        self.assertIsInstance(result, Path)
        self.assertTrue(result.is_absolute())
        self.assertEqual(result, run_root.resolve())

    def test_missing_job_done_fails_when_completion_is_required(self):
        run_root = self.make_run(complete=False)

        with self.assertRaisesRegex(
            FileNotFoundError,
            r"does not appear complete.*job_done\.txt",
        ):
            nnt.validate_run_directory(run_root)

    def test_job_done_must_be_a_regular_file(self):
        run_root = self.make_run(complete=False)
        (run_root / "job_done.txt").mkdir()

        with self.assertRaisesRegex(FileNotFoundError, r"job_done\.txt"):
            nnt.validate_run_directory(run_root)

    def test_missing_job_done_is_accepted_when_completion_is_optional(self):
        run_root = self.make_run(complete=False)

        result = nnt.validate_run_directory(
            run_root,
            require_complete=False,
        )

        self.assertEqual(result, run_root.resolve())

    def test_nested_bias_path_normalizes_to_run_root(self):
        run_root = self.make_run()

        result = nnt.validate_run_directory(run_root / "bias_00000")

        self.assertEqual(result, run_root.resolve())

    def test_same_name_nextnanopy_wrapper_normalizes_to_inner_run_root(self):
        wrapper = self.temporary_root / "wrapped_run"
        nested_run = wrapper / wrapper.name
        nested_run.mkdir(parents=True)
        (nested_run / "job_done.txt").write_text("done\n", encoding="utf-8")
        (nested_run / "bias_00000").mkdir()

        result = nnt.validate_run_directory(wrapper)

        self.assertEqual(result, nested_run.resolve())

    def test_explicitly_selected_existing_bias_is_accepted(self):
        run_root = self.make_run(biases=("bias_00000", "bias_00002"))

        for bias in (2, "2", "bias_00002", "latest"):
            with self.subTest(bias=bias):
                self.assertEqual(
                    nnt.validate_run_directory(run_root, bias=bias),
                    run_root.resolve(),
                )

    def test_missing_selected_bias_raises_clear_error(self):
        run_root = self.make_run(biases=("bias_00000", "bias_00002"))

        with self.assertRaisesRegex(
            FileNotFoundError,
            rf"bias_00001.*{run_root.resolve()}",
        ):
            nnt.validate_run_directory(run_root, bias="bias_00001")

    def test_selected_bias_must_be_a_directory(self):
        run_root = self.make_run(biases=("bias_00001",))
        (run_root / "bias_00000").write_text("not a directory\n", encoding="utf-8")

        with self.assertRaisesRegex(NotADirectoryError, r"bias_00000"):
            nnt.validate_run_directory(run_root, bias="bias_00000")

    def test_bias_none_does_not_require_bias_directory(self):
        run_root = self.make_run(biases=())

        result = nnt.validate_run_directory(run_root, bias=None)

        self.assertEqual(result, run_root.resolve())

    def test_required_output_relative_to_bias_is_accepted(self):
        run_root = self.make_run()
        self.make_output(run_root / "bias_00000", "potential.vtr")

        result = nnt.validate_run_directory(
            run_root,
            required_outputs=("potential.vtr",),
        )

        self.assertEqual(result, run_root.resolve())

    def test_required_output_relative_to_run_root_is_accepted(self):
        run_root = self.make_run()
        self.make_output(run_root, "integrated_density_hole.dat")

        result = nnt.validate_run_directory(
            run_root,
            required_outputs=("integrated_density_hole.dat",),
        )

        self.assertEqual(result, run_root.resolve())

    def test_bias_output_takes_precedence_over_same_run_root_path(self):
        run_root = self.make_run()
        bias_output = self.make_output(
            run_root / "bias_00000",
            "potential.vtr",
            text="bias\n",
        )
        run_output = self.make_output(
            run_root,
            "potential.vtr",
            text="run root\n",
        )
        original_is_file = Path.is_file

        def fail_if_run_duplicate_is_checked(path):
            if path.resolve() == run_output.resolve():
                raise AssertionError("Run-root duplicate was checked after bias match.")
            return original_is_file(path)

        with patch.object(Path, "is_file", new=fail_if_run_duplicate_is_checked):
            result = nnt.validate_run_directory(
                run_root,
                required_outputs=(bias_output.name,),
            )

        self.assertEqual(result, run_root.resolve())

    def test_structure_output_is_accepted_from_run_root(self):
        run_root = self.make_run()
        self.make_output(run_root, "Structure/materials.vtr")

        result = nnt.validate_run_directory(
            run_root,
            required_outputs=("Structure/materials.vtr",),
        )

        self.assertEqual(result, run_root.resolve())

    def test_strain_output_is_accepted_from_run_root(self):
        run_root = self.make_run()
        self.make_output(run_root, "Strain/strain_simulation.vtr")

        result = nnt.validate_run_directory(
            run_root,
            required_outputs=("Strain/strain_simulation.vtr",),
        )

        self.assertEqual(result, run_root.resolve())

    def test_quantum_output_is_accepted_from_selected_bias(self):
        run_root = self.make_run()
        self.make_output(
            run_root / "bias_00000",
            "Quantum/c-Ge_QW/HH/density.vtr",
        )

        result = nnt.validate_run_directory(
            run_root,
            required_outputs=("Quantum/c-Ge_QW/HH/density.vtr",),
        )

        self.assertEqual(result, run_root.resolve())

    def test_absolute_required_output_is_checked_exactly(self):
        run_root = self.make_run()
        absolute_output = self.temporary_root / "shared" / "potential.vtr"
        self.make_output(run_root / "bias_00000", "potential.vtr")

        with self.assertRaisesRegex(
            FileNotFoundError,
            str(absolute_output.resolve()),
        ):
            nnt.validate_run_directory(
                run_root,
                required_outputs=(absolute_output,),
            )

        self.make_output(absolute_output.parent, absolute_output.name)
        self.assertEqual(
            nnt.validate_run_directory(
                run_root,
                required_outputs=(absolute_output,),
            ),
            run_root.resolve(),
        )

    def test_missing_required_output_raises_file_not_found(self):
        run_root = self.make_run()

        with self.assertRaisesRegex(
            FileNotFoundError,
            r"missing\.dat.*bias_00000.*device_run",
        ):
            nnt.validate_run_directory(
                run_root,
                required_outputs=("missing.dat",),
            )

    def test_multiple_missing_outputs_are_listed_in_one_error(self):
        run_root = self.make_run()

        with self.assertRaises(FileNotFoundError) as caught:
            nnt.validate_run_directory(
                run_root,
                required_outputs=("missing_a.dat", "missing_b.vtr"),
            )

        message = str(caught.exception)
        self.assertIn("missing_a.dat", message)
        self.assertIn("missing_b.vtr", message)
        self.assertEqual(message.count("Required outputs are missing"), 1)

    def test_required_output_search_is_not_recursive(self):
        run_root = self.make_run()
        self.make_output(
            run_root / "bias_00000",
            "nested/potential.vtr",
        )
        self.make_output(run_root, "other_nested/potential.vtr")

        with self.assertRaisesRegex(FileNotFoundError, r"potential\.vtr"):
            nnt.validate_run_directory(
                run_root,
                required_outputs=("potential.vtr",),
            )

    def test_required_output_must_be_a_regular_file(self):
        run_root = self.make_run()
        (run_root / "bias_00000" / "potential.vtr").mkdir()

        with self.assertRaisesRegex(FileNotFoundError, r"potential\.vtr"):
            nnt.validate_run_directory(
                run_root,
                required_outputs=("potential.vtr",),
            )

    def test_empty_required_outputs_is_accepted(self):
        run_root = self.make_run()

        result = nnt.validate_run_directory(
            run_root,
            required_outputs=(),
        )

        self.assertEqual(result, run_root.resolve())

    def test_validation_does_not_create_or_modify_filesystem_entries(self):
        run_root = self.make_run()
        self.make_output(run_root / "bias_00000", "potential.vtr")
        before = self.snapshot_tree(self.temporary_root)

        nnt.validate_run_directory(
            run_root,
            required_outputs=("potential.vtr",),
        )

        after = self.snapshot_tree(self.temporary_root)
        self.assertEqual(after, before)

    def test_validation_does_not_import_nextnanopy_or_print(self):
        run_root = self.make_run()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.object(nnt, "_import_nextnanopy") as import_nextnanopy,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = nnt.validate_run_directory(run_root)

        self.assertEqual(result, run_root.resolve())
        import_nextnanopy.assert_not_called()
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_validate_run_directory_is_public(self):
        self.assertIn("validate_run_directory", nnt.__all__)


if __name__ == "__main__":
    unittest.main()
