from ggset import *
from typing import Any, Dict, Optional
import pandas as pd
from pathlib import Path

__all__ = ["dlc_reader"]


class Dlc2021BulkFileResolver(BulkFileResolverStrategy):
    def resolve(self, file: GGFile, bulk_collection: GGBulkCollection) -> GGFile:
        corresponding_file = file.ggdir.get_corresponding_file_for_this_dir("annotations", ".json")
        rel_path = corresponding_file.rel_path.relative_to(bulk_collection.data_root.rel_path)
        return bulk_collection.bulk_files_root.get_file(rel_path)

    def all_files(self, bulk_collection: GGBulkCollection) -> list[GGFile]:
        return list(bulk_collection.bulk_files_root.iterate("annotations", filter_endings=(".json",)))


class Dlc2021BulkStorageStrategy(BulkStorageStrategy):
    def __init__(self):
        self._cache: Dict[str, Dict[str, Dict]] = {}

    def write(self, ref_file: GGFile, bulk_file: GGFile, data: Any, bulk_collection: GGBulkCollection) -> None:
        raise NotImplementedError("This Strategy is only for reading.")

    def read_dataframe(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> pd.DataFrame:
        d = self.read_dict(bulk_file, bulk_collection)
        l = [{"filename": k, **v} for k, v in d.items()]
        return pd.DataFrame(l)

    def read_dict(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> Dict[str, Any]:
        if bulk_file.rel_path.as_posix() in self._cache:
            return self._cache[bulk_file.rel_path.as_posix()]
        data = bulk_file.read_json()
        r = {}
        for value in data["_via_img_metadata"].values():
            name = Path(value["filename"])
            dir = bulk_file.get_corresponding_dir("images")
            p = (
                dir.rel_path.relative_to(bulk_collection.data_root.rel_path) / name
            )  # this is how the keys are expected to be returned, relative to the data root
            r[p.as_posix()] = value
        self._cache[bulk_file.rel_path.as_posix()] = r
        return r

    def get_existing_files_set(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> set[str]:
        return set(self.read_dict(bulk_file, bulk_collection).keys())

    def read_for_file(
        self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: GGBulkCollection
    ) -> Optional[Dict[str, Any]]:
        d = self.read_dict(bulk_file, bulk_collection)
        return d.get(ref_file.rel_path.relative_to(bulk_collection.data_root.rel_path).as_posix(), None)

    def flush(self, bulk_collection: GGBulkCollection) -> None:
        self._cache = {}


def dlc_reader(dlc_set: GGDir) -> GGBulkCollection:
    """Builds a GGBulkCollection for the DLC 2021 dataset structure to access the annotations."""
    return GGBulkCollection(
        dlc_set, bulk_file_resolver_strategy=Dlc2021BulkFileResolver(), storage_strategy=Dlc2021BulkStorageStrategy()
    )
