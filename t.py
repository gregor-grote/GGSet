from src.ggset import GGSet
import os
from pathlib import Path
from zipfile import ZipFile
import shutil
from src.ggset import (
    BulkFileResolverStrategy,
    GGFile,
    GGBulkCollection,
    FixedLayerKeyMappingStrategy,
    JsonStorageStrategy,
)
from typing import List


def print_folder_structure(path: Path, depth: int = 0) -> None:
    for item in path.iterdir():
        if item.is_dir():
            print("  " * depth + item.name + "/")
            print_folder_structure(item, depth + 1)
        else:
            print("  " * depth + item.name)


dataset_path = Path("example/custom_labels_dataset")
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)
with ZipFile(dataset_path.with_suffix(".zip"), "r") as zip_ref:
    zip_ref.extractall(".")

print("Folder structure of the extracted dataset:")
print_folder_structure(dataset_path)
print("\n\n")

custom_labels_dataset = GGSet(dataset_path)


class CustomResolver(BulkFileResolverStrategy):
    def resolve(self, file: GGFile, bulk_collection: GGBulkCollection) -> GGFile:
        print(f"Resolving label for file: {file.rel_path}")
        name = file.ggdir.ancestor_at_level(bulk_collection.bulk_files_root.level + 1).name
        if name == "train":
            return bulk_collection.bulk_files_root.get_file("annotations/train_labels.json")
        elif name == "test":
            return bulk_collection.bulk_files_root.get_file("annotations/test_labels.json")
        else:
            raise ValueError(f"Unknown directory name: {name}")

    def all_files(self, bulk_collection: GGBulkCollection) -> List[GGFile]:
        return [
            bulk_collection.bulk_files_root.get_file("annotations/train_labels.json"),
            bulk_collection.bulk_files_root.get_file("annotations/test_labels.json"),
        ]


with GGBulkCollection(
    custom_labels_dataset,
    bulk_file_resolver_strategy=CustomResolver(),
    storage_strategy=JsonStorageStrategy(FixedLayerKeyMappingStrategy(layer=1)),
) as bulk_collection:
    for file, label in bulk_collection:
        img = file.read_image()
        print(f"File: {file.rel_path}, Average: {img.mean():.2f}, Label: {label}")
