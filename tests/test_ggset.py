from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.ggset.ggset import *


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _create_dataset_1(root: Path) -> None:
    _write(root / "data" / "file1.txt", "1")
    _write(root / "data" / "file2.txt", "2")
    _write(root / "labels" / "file1.csv", "3")
    _write(root / "labels" / "file2.csv", "4")


def _create_dataset_2(root: Path) -> None:
    _write(root / "db1" / "data" / "file1.txt", "1")
    _write(root / "db1" / "data" / "file2.txt", "2")
    _write(root / "db1" / "labels" / "file1.csv", "3")
    _write(root / "db1" / "labels" / "file2.csv", "4")
    _write(root / "db2" / "data" / "file1.txt", "5")
    _write(root / "db2" / "data" / "file2.txt", "6")
    _write(root / "db2" / "labels" / "file1.csv", "7")
    _write(root / "db2" / "labels" / "file2.csv", "8")


def _create_dataset_3(root: Path) -> None:
    _write(root / "train" / "db1" / "data" / "sub_level1" / "file1.txt", "1")
    _write(root / "train" / "db1" / "data" / "sub_level1" / "file2.txt", "2")
    _write(root / "train" / "db1" / "data" / "sub_level2" / "file1.txt", "3")
    _write(root / "train" / "db1" / "data" / "sub_level2" / "file2.txt", "4")
    _write(root / "train" / "db1" / "labels" / "sub_level1" / "file1.txt", "1 2 3")
    _write(root / "train" / "db1" / "labels" / "sub_level1" / "file2.txt", "2 3 4")
    _write(root / "train" / "db1" / "labels" / "sub_level2" / "file1.txt", "3 4 5")
    _write(root / "train" / "db1" / "labels" / "sub_level2" / "file2.txt", "4 5 6")

    _write(root / "train" / "db2" / "data" / "sub_level1" / "file1.txt", "5")
    _write(root / "train" / "db2" / "data" / "sub_level1" / "file2.txt", "6")
    _write(root / "train" / "db2" / "data" / "sub_level2" / "file1.txt", "7")
    _write(root / "train" / "db2" / "data" / "sub_level2" / "file2.txt", "8")

    _write(root / "test" / "db1" / "data" / "sub_level1" / "file1.txt", "9")
    _write(root / "test" / "db1" / "data" / "sub_level1" / "file2.txt", "10")
    _write(root / "test" / "db1" / "data" / "sub_level2" / "file1.txt", "11")
    _write(root / "test" / "db1" / "data" / "sub_level2" / "file2.txt", "12")
    _write(root / "test" / "db1" / "labels" / "sub_level1" / "file1.txt", "9 10 11")
    _write(root / "test" / "db1" / "labels" / "sub_level1" / "file2.txt", "10 11 12")
    _write(root / "test" / "db1" / "labels" / "sub_level2" / "file1.txt", "11 12 13")
    _write(root / "test" / "db1" / "labels" / "sub_level2" / "file2.txt", "12 13 14")

    _write(root / "test" / "db2" / "data" / "sub_level1" / "file1.txt", "13")
    _write(root / "test" / "db2" / "data" / "sub_level1" / "file2.txt", "14")
    _write(root / "test" / "db2" / "data" / "sub_level2" / "file1.txt", "15")
    _write(root / "test" / "db2" / "data" / "sub_level2" / "file2.txt", "16")


class TestGGDir(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.dataset_1_root = Path(self._tmpdir.name) / "dataset_1"
        _create_dataset_1(self.dataset_1_root)
        self.ggset_1 = GGSet(self.dataset_1_root, type_sep_level=1)

        self.dataset_2_root = Path(self._tmpdir.name) / "dataset_2"
        _create_dataset_2(self.dataset_2_root)
        self.ggset_2 = GGSet(self.dataset_2_root, type_sep_level=2)

        self.dataset_3_root = Path(self._tmpdir.name) / "dataset_3"
        _create_dataset_3(self.dataset_3_root)
        self.ggset_3 = GGSet(self.dataset_3_root, type_sep_level=3)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_simple_iteration(self):
        r = list(self.ggset_1.iterate("data"))
        self.assertEqual(len(r), 2)
        self.assertIsInstance(r[0], GGFile)
        file_names = {f.rel_path.name for f in r}
        self.assertEqual(file_names, {"file1.txt", "file2.txt"})

    def test_iterate_with_subdirs(self):
        r = list(self.ggset_2.iterate("data"))
        self.assertEqual(len(r), 4)
        self.assertIsInstance(r[0], GGFile)
        file_names = {f.rel_path.name for f in r}
        self.assertEqual(file_names, {"file1.txt", "file2.txt"})

    def test_iterate_train(self):
        r = list(self.ggset_3.get_sub_dir("train").iterate("data"))
        self.assertEqual(len(r), 8)
        found = False
        for file in r:
            if file.read_text() == "1":
                found = True
                self.assertEqual(file.rel_path.name, "file1.txt")
                self.assertIsInstance(file.read_single_int(), int)
                self.assertEqual(file.read_single_int(), 1)
                self.assertIsInstance(file.read_single_float(), float)
                self.assertEqual(file.read_single_float(), 1.0)

                label = file.get_corresponding_file("labels", ".txt")
                assert label is not None
                self.assertEqual(label.read_text(), "1 2 3")
                self.assertEqual(label.read_int_list(), [1, 2, 3])
        self.assertTrue(found, "Did not find file with content '1' in train data.")
        self.ggset_3.print_tree()

    def test_iterate_all(self):
        r = list(self.ggset_3.iterate())
        self.assertEqual(len(r), 24)
        found = False
        for file in r:
            if file.read_text() == "1":
                found = True
                self.assertEqual(file.rel_path.name, "file1.txt")
                self.assertIsInstance(file.read_single_int(), int)
                self.assertEqual(file.read_single_int(), 1)
                self.assertIsInstance(file.read_single_float(), float)
                self.assertEqual(file.read_single_float(), 1.0)

                label = file.get_corresponding_file("labels", ".txt")
                assert label is not None
                self.assertEqual(label.read_text(), "1 2 3")
                self.assertEqual(label.read_int_list(), [1, 2, 3])
        self.assertTrue(found, "Did not find file with content '1' in data.")

    def test_write_annotation_text(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir"
        _write(small_ggset_root / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, type_sep_level=1)
        file = ggset.get_sub_dir("data").get_file("file1.txt")
        assert file is not None
        file.write_annotation_text("labels", "annotation for file1")
        annotation_file = ggset.get_sub_dir("labels").get_file("file1.txt")
        assert annotation_file is not None
        self.assertIsNotNone(annotation_file, "Annotation file was not created.")
        self.assertEqual(annotation_file.read_text(), "annotation for file1")
        self.assertEqual(annotation_file.rel_path.parent.name, "labels")
        self.assertEqual(annotation_file.rel_path.name, "file1.txt")
        self.assertEqual(len(ggset.sub_dirs), 2)

    def test_write_file(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir"
        _write(small_ggset_root / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, type_sep_level=1)
        file = ggset.get_sub_dir("data").get_file("file1.txt")
        assert file is not None
        cor_dir = file.get_corresponding_dir("labels", force_create=True)
        self.assertEqual(str(cor_dir.rel_path), "labels/file1")
        target_file = cor_dir.get_file("0.txt", force_create=True)
        self.assertEqual(str(target_file.rel_path), "labels/file1/0.txt")
        target_file_2 = cor_dir.write_text_file("eyo")
        self.assertEqual(str(target_file_2.rel_path), "labels/file1/1.txt")
        self.assertEqual(target_file_2.read_text(), "eyo")

    def test_filter_dbs(self):
        self.ggset_3.add_filter_allow_only(2, "db1")
        r = list(self.ggset_3.iterate("data"))
        self.assertEqual(len(r), 8)
        self.ggset_3.filters.clear()

    def test_bulk_csv_writer(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "train" / "data" / "file2.txt", "2")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "3")
        _write(small_ggset_root / "test" / "data" / "file2.txt", "4")

        ggset = GGSet(small_ggset_root, type_sep_level=2)
        bulk_writer = ggset.crate_bulk_csv_writer("bulk_data", layer=2, cols=["col1", "col2"])
        for file in ggset.iterate("data"):
            value = int(file.read_text())
            bulk_writer.write_dict_row(file, {"col1": value, "col2": value * 10})
        bulk_writer.flush()
        # Check that the CSV files were created and contain the correct data
        for sub_dir in ["train", "test"]:
            csv_file = ggset.get_sub_dir(sub_dir).get_file("bulk_data.csv")
            self.assertIsNotNone(csv_file, f"CSV file not found in {sub_dir}")
            assert csv_file is not None
            df = csv_file.read_dataframe()
            expected_values = {"train": {1, 2}, "test": {3, 4}}[sub_dir]
            self.assertEqual(set(df["col1"].tolist()), expected_values)
            self.assertEqual(set(df["col2"].tolist()), {v * 10 for v in expected_values})

        t1_file = ggset.get_sub_dir("train").get_sub_dir("data").get_file("file1.txt")
        assert t1_file is not None
        self.assertEqual(t1_file.read_text(), "1")
        row = bulk_writer.read_row_for_file(t1_file)
        assert row is not None
        self.assertEqual(row["col1"], 1)
        self.assertEqual(row["col2"], 10)

    def test_bulk_csv_on_flat_dataset(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir"
        _write(small_ggset_root / "file1.txt", "1")
        _write(small_ggset_root / "file2.txt", "2")

        ggset = GGSet(small_ggset_root, type_sep_level=-1)
        bulk_writer = ggset.crate_bulk_csv_writer("bulk_data", layer=1, cols=["col1"])
        for file in ggset.iterate():
            value = int(file.read_text())
            bulk_writer.write_dict_row(file, {"col1": value})
        bulk_writer.flush()

        csv_file = ggset.get_file("bulk_data.csv")
        self.assertIsNotNone(csv_file, "CSV file not found in data")
        assert csv_file is not None
        df = csv_file.read_dataframe()
        self.assertEqual(set(df["col1"].tolist()), {1, 2})

    def test_bulk_json_writer(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_json"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "train" / "data" / "file2.txt", "2")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "3")
        _write(small_ggset_root / "test" / "data" / "file2.txt", "4")

        ggset = GGSet(small_ggset_root, type_sep_level=2)
        bulk_writer = ggset.crate_bulk_json_writer("bulk_data", layer=2, cols=["col1", "col2"])
        for file in ggset.iterate("data"):
            value = int(file.read_text())
            bulk_writer.write_dict_row(file, {"col1": value, "col2": value * 10})
        bulk_writer.flush()

        for sub_dir in ["train", "test"]:
            json_file = ggset.get_sub_dir(sub_dir).get_file("bulk_data.json")
            self.assertIsNotNone(json_file, f"JSON file not found in {sub_dir}")
            assert json_file is not None

            payload = json.loads(json_file.read_text())
            self.assertIsInstance(payload, dict)
            self.assertTrue(all(isinstance(k, str) for k in payload.keys()))

            expected_key_prefix = f"{sub_dir}/data/"
            self.assertTrue(all(k.startswith(expected_key_prefix) for k in payload.keys()))

            df = bulk_writer.read_dataframe()
            expected_values = {"train": {1, 2}, "test": {3, 4}}[sub_dir]
            df_sub_dir = df[df["Filename"].str.startswith(f"{sub_dir}/")]
            self.assertEqual(set(df_sub_dir["col1"].tolist()), expected_values)
            self.assertEqual(set(df_sub_dir["col2"].tolist()), {v * 10 for v in expected_values})

        t1_file = ggset.get_sub_dir("train").get_sub_dir("data").get_file("file1.txt")
        assert t1_file is not None
        row = bulk_writer.read_row_for_file(t1_file)
        assert row is not None
        self.assertEqual(row["col1"], 1)
        self.assertEqual(row["col2"], 10)

    def test_bulk_csv_save_rel_paths(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_rel_csv"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "2")

        ggset = GGSet(small_ggset_root, type_sep_level=2)
        bulk_writer = ggset.crate_bulk_csv_writer("bulk_data", layer=2, cols=["col1"], save_rel_paths=True)
        for file in ggset.iterate("data"):
            bulk_writer.write_dict_row(file, {"col1": int(file.read_text())})
        bulk_writer.flush()

        train_csv = ggset.get_sub_dir("train").get_file("bulk_data.csv")
        assert train_csv is not None
        stored_df = train_csv.read_dataframe()
        self.assertEqual(stored_df["Filename"].tolist(), ["data/file1.txt"])

        read_df = bulk_writer.read_dataframe().sort_values("Filename").reset_index(drop=True)
        self.assertEqual(read_df["Filename"].tolist(), ["test/data/file1.txt", "train/data/file1.txt"])

    def test_bulk_json_save_rel_paths(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_rel_json"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "2")

        ggset = GGSet(small_ggset_root, type_sep_level=2)
        bulk_writer = ggset.crate_bulk_json_writer("bulk_data", layer=2, save_rel_paths=True)
        for file in ggset.iterate("data"):
            bulk_writer.write_dict_row(file, {"col1": int(file.read_text())})
        bulk_writer.flush()

        train_json = ggset.get_sub_dir("train").get_file("bulk_data.json")
        assert train_json is not None
        payload = json.loads(train_json.read_text())
        self.assertEqual(list(payload.keys()), ["data/file1.txt"])

        read_df = bulk_writer.read_dataframe().sort_values("Filename").reset_index(drop=True)
        self.assertEqual(read_df["Filename"].tolist(), ["test/data/file1.txt", "train/data/file1.txt"])


if __name__ == "__main__":
    unittest.main()
