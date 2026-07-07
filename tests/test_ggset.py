from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import src.ggset.ggset as ggmod
from src.ggset.ggset import (
    GGSet,
    GGFile,
    GGDir,
    GGBulkCollection,
    BulkFileAtLevelResolverStrategy,
    JsonStorageStrategy,
    DefaultKeyMappingStrategy,
)
import pandas as pd


import numpy as np
import cv2

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
        self.ggset_1 = GGSet(self.dataset_1_root, data_type_level=1)

        self.dataset_2_root = Path(self._tmpdir.name) / "dataset_2"
        _create_dataset_2(self.dataset_2_root)
        self.ggset_2 = GGSet(self.dataset_2_root, data_type_level=2)

        self.dataset_3_root = Path(self._tmpdir.name) / "dataset_3"
        _create_dataset_3(self.dataset_3_root)
        self.ggset_3 = GGSet(self.dataset_3_root, data_type_level=3)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_root_path_is_absolute(self):
        self.assertTrue(self.ggset_1.root_path.is_absolute())
        self.assertEqual(self.ggset_1.rel_path, Path())
        self.assertEqual(self.ggset_1.abs_path, self.ggset_1.root_path)
        self.assertEqual(self.ggset_1.name, self.dataset_1_root.name)

    def test_ggdir_stores_rel_path_from_root(self):
        data_dir = self.ggset_1.get_sub_dir("data")
        self.assertEqual(data_dir.name, "data")
        self.assertEqual(data_dir.rel_path, Path("data"))
        self.assertEqual(data_dir.abs_path, self.ggset_1.root_path / "data")

        file = data_dir.get_file("file1.txt")
        self.assertEqual(file.rel_path, Path("data") / "file1.txt")
        self.assertEqual(file.abs_path, self.ggset_1.root_path / "data" / "file1.txt")

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
                self.assertEqual(label.read_text(), "1 2 3")
                self.assertEqual(label.read_int_list(), [1, 2, 3])
        self.assertTrue(found, "Did not find file with content '1' in data.")

    def test_write_annotation_text(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir"
        _write(small_ggset_root / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        file = ggset.get_sub_dir("data").get_file("file1.txt")
        file.get_corresponding_file("labels", ".txt").write_text("annotation for file1")
        annotation_file = ggset.get_sub_dir("labels").get_file("file1.txt")
        self.assertTrue(annotation_file.exists(), "Annotation file was not created.")
        self.assertEqual(annotation_file.read_text(), "annotation for file1")
        self.assertEqual(annotation_file.rel_path.parent.name, "labels")
        self.assertEqual(annotation_file.rel_path.name, "file1.txt")
        self.assertEqual(len(ggset.sub_dirs), 2)

    def test_write_file(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir"
        _write(small_ggset_root / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        file = ggset.get_sub_dir("data").get_file("file1.txt")
        cor_dir = file.get_corresponding_dir("labels")
        self.assertFalse(cor_dir.exists())
        self.assertEqual(str(cor_dir.rel_path), "labels/file1")
        target_file = cor_dir.get_file("0.txt")
        self.assertFalse(target_file.exists())
        self.assertEqual(str(target_file.rel_path), "labels/file1/0.txt")
        target_file_2 = cor_dir.get_new_sub_file(".txt")
        self.assertFalse(target_file_2.exists())
        target_file_2.write_text("eyo")
        self.assertEqual(str(target_file_2.rel_path), "labels/file1/1.txt")
        self.assertEqual(target_file_2.read_text(), "eyo")

    def test_filter_dbs(self):
        self.ggset_3.add_filter_allow_only(2, "db1")
        r = list(self.ggset_3.iterate("data"))
        self.assertEqual(len(r), 8)
        self.ggset_3.filters.clear()

    def test_bulk_json_writer(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_json"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "train" / "data" / "file2.txt", "2")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "3")
        _write(small_ggset_root / "test" / "data" / "file2.txt", "4")

        ggset = GGSet(small_ggset_root, data_type_level=2)
        bulk_writer = ggset.create_bulk_json_collection("bulk_data.json", layer=2)
        for file in ggset.iterate("data"):
            value = int(file.read_text())
            bulk_writer.write(file, {"col1": value, "col2": value * 10})
        bulk_writer.flush()

        for sub_dir in ["train", "test"]:
            json_file = ggset.get_sub_dir(sub_dir).get_file("bulk_data.json")
            self.assertTrue(json_file.exists(), f"JSON file not found in {sub_dir}")

            payload = json.loads(json_file.read_text())
            self.assertIsInstance(payload, dict)
            self.assertTrue(all(isinstance(k, str) for k in payload.keys()))

            expected_key_prefix = f"{sub_dir}/data/"
            self.assertTrue(all(k.startswith(expected_key_prefix) for k in payload.keys()))

            df = bulk_writer.read_dataframe()
            expected_values = {"train": {1, 2}, "test": {3, 4}}[sub_dir]
            df_sub_dir = df[df["filename"].str.startswith(f"{sub_dir}/")]
            self.assertEqual(set(df_sub_dir["col1"].tolist()), expected_values)
            self.assertEqual(set(df_sub_dir["col2"].tolist()), {v * 10 for v in expected_values})

        t1_file = ggset.get_sub_dir("train").get_sub_dir("data").get_file("file1.txt")
        row = bulk_writer.read_for_file(t1_file)
        assert row is not None
        self.assertEqual(row["col1"], 1)
        self.assertEqual(row["col2"], 10)

    def test_bulk_json_save_rel_paths(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_rel_json"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "2")

        ggset = GGSet(small_ggset_root, data_type_level=2)
        bulk_writer = ggset.create_bulk_json_collection("bulk_data.json", layer=2, rel_paths=True)
        for file in ggset.iterate("data"):
            bulk_writer.write(file, {"col1": int(file.read_text())})
        bulk_writer.flush()

        train_json = ggset.get_sub_dir("train").get_file("bulk_data.json")
        self.assertTrue(train_json.exists())
        payload = json.loads(train_json.read_text())
        self.assertEqual(list(payload.keys()), ["data/file1.txt"])

        read_df = bulk_writer.read_dataframe().sort_values("filename").reset_index(drop=True)
        self.assertEqual(read_df["filename"].tolist(), ["test/data/file1.txt", "train/data/file1.txt"])

    def test_write_text_from_dir(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_write_from_dir"
        _write(small_ggset_root / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        data_dir = ggset.get_sub_dir("data")
        new_file = data_dir.get_file("new_file.txt")
        self.assertFalse(new_file.exists())
        new_file.write_text("new content")
        self.assertTrue(new_file.exists())
        self.assertEqual(new_file.read_text(), "new content")
        self.assertEqual(new_file.rel_path.name, "new_file.txt")
        self.assertEqual(new_file.rel_path.parent.name, "data")

    def test_write_image_from_dir(self):

        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_write_image"
        _write(small_ggset_root / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        data_dir = ggset.get_sub_dir("data")

        # Create a simple image using numpy
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[0:5, 0:5] = [255, 0, 0]  # Red square in top-left
        image[5:10, 5:10] = [0, 255, 0]  # Green square in bottom-right

        new_file = data_dir.get_file("test_image.png")
        self.assertFalse(new_file.exists())
        new_file.write_image(image)
        self.assertTrue(new_file.exists())
        self.assertEqual(new_file.rel_path.name, "test_image.png")
        self.assertEqual(new_file.rel_path.parent.name, "data")

        # Read the image back and check its content
        read_image = cv2.imread(str(new_file.abs_path))
        self.assertIsNotNone(read_image)
        assert read_image is not None
        self.assertTrue(np.array_equal(image, read_image), "The written and read images do not match.")

    def test_get_file_for_dir(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_get_file_for_dir"
        _write(small_ggset_root / "data" / "db1" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        data_dir = ggset.get_sub_dir("data")
        file_for_dir = data_dir.get_corresponding_file_for_this_dir("labels", ".txt")
        self.assertFalse(file_for_dir.exists())
        self.assertEqual(file_for_dir.rel_path.name, "data.txt")
        self.assertEqual(file_for_dir.rel_path.parent.name, "labels")
        file_for_dir.write_text("annotation for data dir")
        annotation_file = ggset.get_sub_dir("labels").get_file("data.txt")
        self.assertTrue(annotation_file.exists(), "Annotation file for data dir was not created.")
        self.assertEqual(annotation_file.read_text(), "annotation for data dir")

    def test_get_dir_with_slash_in_name(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_get_dir_with_slash"
        _write(small_ggset_root / "data" / "db1" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        sub_dir = ggset.get_sub_dir("data/db3")
        self.assertEqual(sub_dir.rel_path, Path("data/db3"))
        self.assertFalse(sub_dir.exists())
        self.assertFalse(ggset.get_sub_dir("data/db4").exists())

    def test_get_file_with_slash_in_name(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_get_file_with_slash"
        _write(small_ggset_root / "data" / "db1" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        file = ggset.get_file("data/db1/file1.txt")
        self.assertTrue(file.exists())
        self.assertEqual(file.rel_path, Path("data/db1/file1.txt"))
        nen_existing_file = ggset.get_file("data/db1/non_existing.txt")
        self.assertFalse(nen_existing_file.exists())

    def test_add_data_in_json(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_add_data_json"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        ggset = GGSet(small_ggset_root, data_type_level=1)
        bulk_writer = ggset.create_bulk_json_collection("bulk_data.json", layer=1)
        file = ggset.get_file("train/data/file1.txt")
        self.assertTrue(file.exists())
        bulk_writer.write(file, {"a": 1})
        bulk_writer.flush()
        json_file = ggset.get_file("bulk_data.json")
        self.assertTrue(json_file.exists())
        payload = json.loads(json_file.read_text())
        self.assertIsInstance(payload, dict)
        self.assertIn("train/data/file1.txt", payload)
        self.assertEqual(payload["train/data/file1.txt"]["a"], 1)

        bulk_writer.write(file, {"b": 2})
        bulk_writer.flush()
        payload_updated = json.loads(json_file.read_text())
        self.assertIsInstance(payload_updated, dict)
        self.assertIn("train/data/file1.txt", payload_updated)
        self.assertEqual(payload_updated["train/data/file1.txt"]["a"], 1)
        self.assertEqual(payload_updated["train/data/file1.txt"]["b"], 2)

        bulk_writer.write(file, {"a": 10})
        bulk_writer.flush()
        payload_overwrite = json.loads(json_file.read_text())
        self.assertIsInstance(payload_overwrite, dict)
        self.assertIn("train/data/file1.txt", payload_overwrite)
        self.assertEqual(payload_overwrite["train/data/file1.txt"]["a"], 10)
        self.assertEqual(payload_overwrite["train/data/file1.txt"]["b"], 2)

    def test_iterate_filter_ending(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_filter_ending"
        _write(small_ggset_root / "data" / "dir" / "file1.txt", "1")
        _write(small_ggset_root / "data" / "file2.csv", "2")
        ggset = GGSet(small_ggset_root, data_type_level=-1)
        r = list()
        for item in ggset.iterate(filter_endings=(".txt",)):
            r.append(item)
        self.assertEqual(len(r), 1)
        self.assertEqual(r[0].rel_path.name, "file1.txt")

    def test_file_count(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_file_count"
        _write(small_ggset_root / "data" / "dir" / "file1.txt", "1")
        _write(small_ggset_root / "data" / "file2.csv", "2")
        ggset = GGSet(small_ggset_root, data_type_level=-1)
        self.assertEqual(ggset.file_count(), 2)
        self.assertEqual(ggset.file_count(filter_endings=(".txt",)), 1)
        self.assertEqual(ggset.file_count(filter_endings=(".csv",)), 1)
        self.assertEqual(ggset.file_count(filter_endings=(".json",)), 0)

    def test_dynamic_csv_writer(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_dynamic_csv"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file1.txt", "2")

        ggset = GGSet(small_ggset_root, data_type_level=2)
        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2, rel_paths=True) as bulk_writer:
            for file in ggset.iterate("data"):
                text = file.read_text()
                if text == "1":
                    bulk_writer.write(file, {"col1": 1, "col2": 10})
                elif text == "2":
                    bulk_writer.write(file, {"col3": 2, "col4": 20})

        bulk_data_csv_train = ggset.get_file("train/bulk_data.csv")
        self.assertTrue(bulk_data_csv_train.exists())
        df_train = bulk_data_csv_train.read_dataframe()
        self.assertEqual(set(df_train.columns), {"filename", "col1", "col2"})
        self.assertEqual(df_train["filename"].tolist(), ["data/file1.txt"])
        self.assertEqual(df_train["col1"].tolist(), [1])
        self.assertEqual(df_train["col2"].tolist(), [10])

        bulk_data_csv_test = ggset.get_file("test/bulk_data.csv")
        self.assertTrue(bulk_data_csv_test.exists())
        df_test = bulk_data_csv_test.read_dataframe()
        self.assertEqual(set(df_test.columns), {"filename", "col3", "col4"})
        self.assertEqual(df_test["filename"].tolist(), ["data/file1.txt"])
        self.assertEqual(df_test["col3"].tolist(), [2])
        self.assertEqual(df_test["col4"].tolist(), [20])

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2, rel_paths=True) as bulk_writer:
            df = bulk_writer.read_dataframe()

            self.assertEqual(set(df.columns), {"filename", "col1", "col2", "col3", "col4"})
            self.assertEqual(df["filename"].tolist(), ["test/data/file1.txt", "train/data/file1.txt"])
            train_row = df[df["filename"] == "train/data/file1.txt"].iloc[0]
            self.assertEqual(train_row["col1"], 1)
            self.assertEqual(train_row["col2"], 10)
            self.assertTrue(pd.isna(train_row["col3"]))
            self.assertTrue(pd.isna(train_row["col4"]))

            test_row = df[df["filename"] == "test/data/file1.txt"].iloc[0]
            self.assertTrue(pd.isna(test_row["col1"]))
            self.assertTrue(pd.isna(test_row["col2"]))
            self.assertEqual(test_row["col3"], 2)
            self.assertEqual(test_row["col4"], 20)

    def test_csv_reader_in_sub_dir(self):
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_csv_reader_subdir"
        _write(small_ggset_root / "train" / "file1.txt", "1")
        _write(small_ggset_root / "train" / "bulk_data.csv", "filename,col1\nfile1.txt,3")
        ggset = GGSet(small_ggset_root)
        with ggset.get_sub_dir("train").create_bulk_csv_collection(
            "bulk_data.csv", 2, rel_paths=True, caching=True
        ) as bulk_reader:
            for d in bulk_reader:
                print(d)
        bulk_file = ggset.get_sub_dir("train").get_file("bulk_data.csv")
        self.assertTrue(bulk_file.exists())
        df = bulk_file.read_dataframe()
        self.assertEqual(df["filename"].tolist(), ["file1.txt"])
        self.assertEqual(df["col1"].tolist(), [3])

    # ------------------------------------------------------------------
    # Tests for target_set parameter on get_corresponding_* methods
    # ------------------------------------------------------------------

    def test_get_corresponding_file_in_target_set(self):
        """get_corresponding_file with target_set resolves into the alternate set."""
        source_root = Path(self._tmpdir.name) / "src_corr_file"
        target_root = Path(self._tmpdir.name) / "tgt_corr_file"
        source_root_resolved = source_root.resolve()
        target_root_resolved = target_root.resolve()
        _write(source_root / "data" / "file1.txt", "1")
        source = GGSet(source_root, data_type_level=1)
        target = GGSet(target_root, data_type_level=1)

        src_file = source.get_sub_dir("data").get_file("file1.txt")
        result = src_file.get_corresponding_file("labels", ".txt", target_set=target)

        # Path must be in the target set, not the source set
        self.assertTrue(result.abs_path.is_relative_to(target_root_resolved))
        self.assertFalse(result.abs_path.is_relative_to(source_root_resolved))
        self.assertEqual(result.rel_path, Path("labels") / "file1.txt")
        self.assertFalse(result.exists())

        # Writing should create the file in the target set
        result.write_text("annotation")
        self.assertTrue(result.exists())
        self.assertFalse((source_root_resolved / "labels" / "file1.txt").exists())

    def test_get_corresponding_dir_in_target_set(self):
        """get_corresponding_dir with target_set resolves into the alternate set."""
        source_root = Path(self._tmpdir.name) / "src_corr_dir"
        target_root = Path(self._tmpdir.name) / "tgt_corr_dir"
        source_root_resolved = source_root.resolve()
        target_root_resolved = target_root.resolve()
        _write(source_root / "data" / "file1.txt", "1")
        source = GGSet(source_root, data_type_level=1)
        target = GGSet(target_root, data_type_level=1)

        src_file = source.get_sub_dir("data").get_file("file1.txt")
        result = src_file.get_corresponding_dir("labels", target_set=target)

        self.assertTrue(result.abs_path.is_relative_to(target_root_resolved))
        self.assertFalse(result.abs_path.is_relative_to(source_root_resolved))
        self.assertEqual(result.rel_path, Path("labels") / "file1")
        self.assertFalse(result.exists())

    def test_get_corresponding_file_in_same_dir_in_target_set(self):
        """get_corresponding_file_in_same_dir with target_set resolves into the alternate set."""
        source_root = Path(self._tmpdir.name) / "src_same_dir"
        target_root = Path(self._tmpdir.name) / "tgt_same_dir"
        source_root_resolved = source_root.resolve()
        target_root_resolved = target_root.resolve()
        _write(source_root / "data" / "file1.txt", "1")
        source = GGSet(source_root)
        target = GGSet(target_root)

        src_file = source.get_sub_dir("data").get_file("file1.txt")
        result = src_file.get_corresponding_file_in_same_dir(".csv", target_set=target)

        self.assertTrue(result.abs_path.is_relative_to(target_root_resolved))
        self.assertFalse(result.abs_path.is_relative_to(source_root_resolved))
        self.assertEqual(result.rel_path, Path("data") / "file1.csv")
        self.assertFalse(result.exists())

        result.write_text("a,b\n1,2")
        self.assertTrue(result.exists())
        self.assertFalse((source_root_resolved / "data" / "file1.csv").exists())

    def test_get_corresponding_dir_in_same_dir_in_target_set(self):
        """get_corresponding_dir_in_same_dir with target_set resolves into the alternate set."""
        source_root = Path(self._tmpdir.name) / "src_same_dir2"
        target_root = Path(self._tmpdir.name) / "tgt_same_dir2"
        source_root_resolved = source_root.resolve()
        target_root_resolved = target_root.resolve()
        _write(source_root / "data" / "file1.txt", "1")
        source = GGSet(source_root)
        target = GGSet(target_root)

        src_file = source.get_sub_dir("data").get_file("file1.txt")
        result = src_file.get_corresponding_dir_in_same_dir(target_set=target)

        self.assertTrue(result.abs_path.is_relative_to(target_root_resolved))
        self.assertFalse(result.abs_path.is_relative_to(source_root_resolved))
        self.assertEqual(result.rel_path, Path("data") / "file1")
        self.assertFalse(result.exists())

    def test_get_corresponding_file_for_this_dir_in_target_set(self):
        """get_corresponding_file_for_this_dir with target_set resolves into the alternate set."""
        source_root = Path(self._tmpdir.name) / "src_file_for_dir"
        target_root = Path(self._tmpdir.name) / "tgt_file_for_dir"
        source_root_resolved = source_root.resolve()
        target_root_resolved = target_root.resolve()
        _write(source_root / "data" / "file1.txt", "1")
        source = GGSet(source_root, data_type_level=1)
        target = GGSet(target_root, data_type_level=1)

        data_dir = source.get_sub_dir("data")
        result = data_dir.get_corresponding_file_for_this_dir("labels", ".txt", target_set=target)

        self.assertTrue(result.abs_path.is_relative_to(target_root_resolved))
        self.assertFalse(result.abs_path.is_relative_to(source_root_resolved))
        self.assertEqual(result.rel_path, Path("labels") / "data.txt")
        self.assertFalse(result.exists())

        result.write_text("dir annotation")
        self.assertTrue(result.exists())
        self.assertFalse((source_root_resolved / "labels" / "data.txt").exists())

    def test_get_corresponding_file_in_target_set_nested(self):
        """target_set works correctly with data_type_level > 1 (nested branches)."""
        source_root = Path(self._tmpdir.name) / "src_nested"
        target_root = Path(self._tmpdir.name) / "tgt_nested"
        target_root_resolved = target_root.resolve()
        source_root_resolved = source_root.resolve()
        _write(source_root / "db1" / "data" / "file1.txt", "1")
        source = GGSet(source_root, data_type_level=2)
        target = GGSet(target_root, data_type_level=2)

        src_file = source.get_sub_dir("db1").get_sub_dir("data").get_file("file1.txt")
        result = src_file.get_corresponding_file("labels", ".csv", target_set=target)

        self.assertTrue(result.abs_path.is_relative_to(target_root_resolved))
        self.assertEqual(result.rel_path, Path("db1") / "labels" / "file1.csv")
        self.assertFalse(result.exists())

        result.write_text("x,y\n1,2")
        self.assertTrue(result.exists())
        self.assertFalse((source_root_resolved / "db1" / "labels" / "file1.csv").exists())

    def test_bulk_csv_write_root(self):
        """CSV bulk collection with write_root writes annotations to a separate GGSet."""
        source_root = Path(self._tmpdir.name) / "src_bulk_csv"
        annot_root = Path(self._tmpdir.name) / "annot_bulk_csv"
        source_root_resolved = source_root.resolve()
        annot_root_resolved = annot_root.resolve()
        _write(source_root / "train" / "data" / "file1.txt", "1")
        _write(source_root / "train" / "data" / "file2.txt", "2")
        _write(source_root / "test" / "data" / "file1.txt", "3")

        source = GGSet(source_root, data_type_level=2)
        annot = GGSet(annot_root, data_type_level=2)

        with source.create_bulk_csv_collection("annot.csv", layer=2, bulk_files_root=annot) as bulk:
            for file in source.iterate("data"):
                bulk.write(file, {"score": int(file.read_text())})

        # CSV files must be in the annotation set, not the source set
        self.assertTrue((annot_root_resolved / "train" / "annot.csv").exists())
        self.assertTrue((annot_root_resolved / "test" / "annot.csv").exists())
        self.assertFalse((source_root_resolved / "train" / "annot.csv").exists())
        self.assertFalse((source_root_resolved / "test" / "annot.csv").exists())

        # iter() must yield GGFiles from the source set
        with source.create_bulk_csv_collection("annot.csv", layer=2, bulk_files_root=annot) as bulk:
            items = list(bulk)
        self.assertEqual(len(items), 3)
        for src_file, data in items:
            self.assertTrue(src_file.abs_path.is_relative_to(source_root_resolved))
            self.assertIn("score", data)

    def test_bulk_json_write_root(self):
        """JSON bulk collection with write_root writes annotations to a separate GGSet."""
        source_root = Path(self._tmpdir.name) / "src_bulk_json"
        annot_root = Path(self._tmpdir.name) / "annot_bulk_json"
        source_root_resolved = source_root.resolve()
        annot_root_resolved = annot_root.resolve()
        _write(source_root / "train" / "data" / "file1.txt", "10")
        _write(source_root / "test" / "data" / "file1.txt", "20")

        source = GGSet(source_root, data_type_level=2)
        annot = GGSet(annot_root, data_type_level=2)

        bulk = source.create_bulk_json_collection("annot.json", layer=2, bulk_files_root=annot)
        for file in source.iterate("data"):
            bulk.write(file, {"value": int(file.read_text())})
        bulk.flush()

        # JSON files must be in the annotation set, not the source set
        self.assertTrue((annot_root_resolved / "train" / "annot.json").exists())
        self.assertTrue((annot_root_resolved / "test" / "annot.json").exists())
        self.assertFalse((source_root_resolved / "train" / "annot.json").exists())
        self.assertFalse((source_root_resolved / "test" / "annot.json").exists())

        # read_for_file must resolve the annotation from write_root
        train_file = source.get_sub_dir("train").get_sub_dir("data").get_file("file1.txt")
        row = bulk.read_for_file(train_file)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["value"], 10)

        # iter() must yield GGFiles from the source set
        items = list(bulk)
        self.assertEqual(len(items), 2)
        for src_file, data in items:
            self.assertTrue(src_file.abs_path.is_relative_to(source_root_resolved))
            self.assertIn("value", data)

    def test_bulk_csv_file_is_lazy_created_only_on_write(self):
        """CSV bulk wrapper is always returned (never None) and file is created only after data is written."""
        root = Path(self._tmpdir.name) / "bulk_lazy_csv"
        _write(root / "train" / "data" / "file1.txt", "1")
        ggset = GGSet(root, data_type_level=2)

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2) as bulk:
            train_dir = ggset.get_sub_dir("train")
            bulk_file = train_dir.get_file("bulk_data.csv")
            self.assertFalse(bulk_file.exists())

            # read operations on a not-yet-existing file must return empty/None
            data_file = ggset.get_file("train/data/file1.txt")
            self.assertIsNone(bulk.read_for_file(data_file))
            self.assertTrue(bulk.read_dataframe().empty)
            self.assertEqual(bulk.read_dict(), {})
            self.assertEqual(bulk.get_existing_files_set(), set())

        # No rows were written, so the file must still not exist.
        self.assertFalse(ggset.get_file("train/bulk_data.csv").exists())

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2) as bulk:
            data_file = ggset.get_file("train/data/file1.txt")
            bulk.write(data_file, {"v": 1})

        self.assertTrue(ggset.get_file("train/bulk_data.csv").exists())

    def test_bulk_csv_caching_file_is_lazy_created_only_on_write(self):
        """Caching CSV bulk wrapper is always returned (never None) and file is created only after data is written."""
        root = Path(self._tmpdir.name) / "bulk_lazy_csv_caching"
        _write(root / "train" / "data" / "file1.txt", "1")
        ggset = GGSet(root, data_type_level=2)

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2, caching=True) as bulk:
            train_dir = ggset.get_sub_dir("train")
            bulk_file = train_dir.get_file("bulk_data.csv")
            self.assertFalse(bulk_file.exists())

            # read operations must return empty/None before file is written
            data_file = ggset.get_file("train/data/file1.txt")
            self.assertIsNone(bulk.read_for_file(data_file))
            self.assertTrue(bulk.read_dataframe().empty)
            self.assertEqual(bulk.read_dict(), {})
            self.assertEqual(bulk.get_existing_files_set(), set())

        # No rows written → file must not exist
        self.assertFalse(ggset.get_file("train/bulk_data.csv").exists())

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2, caching=True) as bulk:
            data_file = ggset.get_file("train/data/file1.txt")
            bulk.write(data_file, {"v": 1})

        self.assertTrue(ggset.get_file("train/bulk_data.csv").exists())

    def test_bulk_json_file_is_lazy_created_only_on_write(self):
        """JSON bulk wrapper is always returned (never None) and file is created only after data is written."""
        root = Path(self._tmpdir.name) / "bulk_lazy_json"
        _write(root / "train" / "data" / "file1.txt", "1")
        ggset = GGSet(root, data_type_level=2)

        with ggset.create_bulk_json_collection("bulk_data.json", layer=2) as bulk:
            train_dir = ggset.get_sub_dir("train")
            bulk_file = train_dir.get_file("bulk_data.json")
            self.assertIsNotNone(bulk_file)
            self.assertFalse(bulk_file.exists())

            # read operations before any write must return empty/None
            data_file = ggset.get_file("train/data/file1.txt")
            self.assertIsNone(bulk.read_for_file(data_file))
            self.assertEqual(bulk.read_dict(), {})
            self.assertEqual(bulk.get_existing_files_set(), set())

        # No rows were written, so the file must still not exist.
        self.assertFalse(ggset.get_file("train/bulk_data.json").exists())

        with ggset.create_bulk_json_collection("bulk_data.json", layer=2) as bulk:
            data_file = ggset.get_file("train/data/file1.txt")
            bulk.write(data_file, {"v": 1})

        self.assertTrue(ggset.get_file("train/bulk_data.json").exists())

    def test_set_root_to_different_level_csv(self):
        """Test that set_root can change the root of a GGSet to a different level."""
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_set_root"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file2.txt", "2")
        ggset = GGSet(small_ggset_root, data_type_level=2)

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2, rel_paths=True) as bulk:
            for file in ggset.iterate("data"):
                bulk.write(file, {"value": int(file.read_text())})

        train_annot_file = ggset.get_file("train/bulk_data.csv")
        self.assertTrue(train_annot_file.exists())
        train_df = train_annot_file.read_dataframe()
        self.assertEqual(set(train_df.columns), {"filename", "value"})
        self.assertEqual(train_df["filename"].tolist(), ["data/file1.txt"])

        test_annot_file = ggset.get_file("test/bulk_data.csv")
        self.assertTrue(test_annot_file.exists())
        test_df = test_annot_file.read_dataframe()
        self.assertEqual(set(test_df.columns), {"filename", "value"})
        self.assertEqual(test_df["filename"].tolist(), ["data/file2.txt"])

        with ggset.get_sub_dir("train").create_bulk_csv_collection(
            "bulk_data.csv", layer=1, rel_paths=True, caching=True
        ) as train_bulk:
            df = train_bulk.read_dataframe()
            self.assertEqual(set(df.columns), {"filename", "value"})
            self.assertEqual(df["filename"].tolist(), ["data/file1.txt"])
            train_bulk.write(ggset.get_file("train/data/file1.txt"), {"value2": 100})

        train_df_updated = ggset.get_file("train/bulk_data.csv").read_dataframe()
        self.assertEqual(set(train_df_updated.columns), {"filename", "value", "value2"})
        self.assertEqual(train_df_updated["filename"].tolist(), ["data/file1.txt"])

    def test_set_root_to_different_level_csv_non_caching(self):
        """Test that set_root can change the root of a GGSet to a different level."""
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_set_root"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file2.txt", "2")
        ggset = GGSet(small_ggset_root, data_type_level=2)

        with ggset.create_bulk_csv_collection("bulk_data.csv", layer=2, rel_paths=True) as bulk:
            for file in ggset.iterate("data"):
                bulk.write(file, {"value": int(file.read_text())})

        train_annot_file = ggset.get_file("train/bulk_data.csv")
        self.assertTrue(train_annot_file.exists())
        train_df = train_annot_file.read_dataframe()
        self.assertEqual(set(train_df.columns), {"filename", "value"})
        self.assertEqual(train_df["filename"].tolist(), ["data/file1.txt"])

        test_annot_file = ggset.get_file("test/bulk_data.csv")
        self.assertTrue(test_annot_file.exists())
        test_df = test_annot_file.read_dataframe()
        self.assertEqual(set(test_df.columns), {"filename", "value"})
        self.assertEqual(test_df["filename"].tolist(), ["data/file2.txt"])

        with ggset.get_sub_dir("train").create_bulk_csv_collection(
            "bulk_data.csv", layer=1, rel_paths=True, caching=True
        ) as train_bulk:
            df = train_bulk.read_dataframe()
            self.assertEqual(set(df.columns), {"filename", "value"})
            self.assertEqual(df["filename"].tolist(), ["data/file1.txt"])

    def test_set_root_to_different_level_json(self):
        """Test that set_root can change the root of a GGSet to a different level."""
        small_ggset_root = Path(self._tmpdir.name) / "small_GGDir_set_root"
        _write(small_ggset_root / "train" / "data" / "file1.txt", "1")
        _write(small_ggset_root / "test" / "data" / "file2.txt", "2")
        ggset = GGSet(small_ggset_root, data_type_level=2)

        with ggset.create_bulk_json_collection("bulk_data.json", layer=2, rel_paths=True) as bulk:
            for file in ggset.iterate("data"):
                bulk.write(file, {"value": int(file.read_text())})

        train_annot_file = ggset.get_file("train/bulk_data.json")
        self.assertTrue(train_annot_file.exists())
        train_dict = train_annot_file.read_json()
        self.assertEqual(set(train_dict.keys()), {"data/file1.txt"})

        test_annot_file = ggset.get_file("test/bulk_data.json")
        self.assertTrue(test_annot_file.exists())
        test_dict = test_annot_file.read_json()
        self.assertEqual(set(test_dict.keys()), {"data/file2.txt"})

        with ggset.get_sub_dir("train").create_bulk_json_collection(
            "bulk_data.json", layer=1, rel_paths=True
        ) as train_bulk:
            df = train_bulk.read_dataframe()
            self.assertEqual(set(df.columns), {"filename", "value"})
            self.assertEqual(df["filename"].tolist(), ["data/file1.txt"])
            train_bulk.write(ggset.get_file("train/data/file1.txt"), {"value2": 100})

        train_dict_updated = ggset.get_file("train/bulk_data.json").read_json()
        self.assertEqual(set(train_dict_updated.keys()), {"data/file1.txt"})
        data = train_dict_updated["data/file1.txt"]
        self.assertEqual(data["value"], 1)
        self.assertEqual(data["value2"], 100)


if __name__ == "__main__":
    unittest.main()
