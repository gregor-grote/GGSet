"""Utilities for navigating hierarchical dataset folders.

This module provides a suite of classes for structured dataset management:

- ``GGSet`` / ``GGDir`` to model hierarchical directory trees, apply level-based
    filtering, and cleanly iterate through files or map counterpart branches.
- ``GGFile`` to represent a concrete file and offer typed convenience
    readers/writers (images, dataframes, text, numbers, JSON, YAML, NumPy arrays).
- Bulk Collection Classes (``GGBulkJsonFileCollection``, ``GGBulkCsvFileCollection``)
    to efficiently manage aggregate metadata and annotations across dataset layers.
"""

from __future__ import annotations

from io import TextIOWrapper
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Generator,
    Self,
    Set,
    Tuple,
    Union,
    TypeVar,
    Generic,
)
import os
from pathlib import Path
import cv2
import json
import yaml
import csv
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np

__all__ = [
    "GGDir",
    "GGFile",
    "GGSet",
    "GGNotFoundError",
    "GGDirNotFoundError",
    "GGFileNotFoundError",
    "GGBulkCollection",
]


class GGNotFoundError(ValueError):
    """Raised when a requested directory or file is not found in the GGDir tree."""


class GGDirNotFoundError(GGNotFoundError):
    """Raised when a requested directory is not found in the GGDir tree."""


class GGFileNotFoundError(GGNotFoundError):
    """Raised when a requested file is not found in the GGDir tree."""


class GGDir:
    """Represents a directory node within a hierarchical dataset tree structure."""

    def __init__(
        self,
        path: str | Path,
        parent: GGDir | None = None,
        data_type_level: int = -1,
        level: int = 0,
    ) -> None:
        """Initialize a GGDir node.
        Children are not populated until ``sub_dirs`` is accessed, allowing for lazy loading of the directory tree.

        Args:
            path: Directory represented by this node.
            parent: Parent node, or ``None`` for the root node.
            data_type_level: Remaining depth until type-separation nodes are
                reached (``0`` means this node is type-separation level, ``-1`` means no type-separation).
            level: Depth from the root node.
        """
        assert level != 0 or isinstance(
            self, GGSet
        ), "Only the root node can have level 0, and it must be an instance of GGSet."
        input_path = Path(path)
        self.parent = parent
        self._sub_dirs: Optional[List[GGDir]] = None
        self.data_type_level = data_type_level
        self.level = level
        if self.parent is None:
            self.rel_path = Path()
        else:
            self.rel_path = input_path
        self._build()

    @property
    def name(self) -> str:
        """Return the directory name derived from its path."""
        if self.parent is None:
            return self.abs_path.name
        return self.rel_path.name

    @property
    def abs_path(self) -> Path:
        """Return the absolute filesystem path for this node."""
        return self.root.root_path / self.rel_path

    @property
    def sub_dirs(self) -> List[GGDir]:
        """Return the list of subdirectory nodes."""
        if self._sub_dirs is None:
            return self._build()
        return self._sub_dirs

    def _build(self) -> List[GGDir]:
        """Populate ``sub_dirs`` with subdirectory nodes."""
        if not self.abs_path.exists():
            return []
        if not self.abs_path.is_dir():
            raise ValueError(f"Expected '{self.abs_path}' to be a directory, but it is a file.")
        for item in self.abs_path.iterdir():
            if item.is_dir():
                child = GGDir(
                    self.rel_path / item.name,
                    parent=self,
                    data_type_level=self.data_type_level,
                    level=self.level + 1,
                )
                if self._sub_dirs is None:
                    self._sub_dirs = []
                self._sub_dirs.append(child)
        if self._sub_dirs is None:
            self._sub_dirs = []
        return self._sub_dirs

    def get_sub_dir(self, name: str | Path) -> GGDir:
        """Return a direct child node by name.

        Args:
            name: Child directory name. Supports forward-slash separated paths.
        """
        if isinstance(name, Path):
            name = name.as_posix()
        if "/" in name:
            name_parts = name.split("/")
            name = name_parts[0]
            further_parts = "/".join(name_parts[1:])
        else:
            further_parts = None
        for sub_dir in self.sub_dirs:
            if sub_dir.name == name:
                if further_parts:
                    return sub_dir.get_sub_dir(further_parts)
                return sub_dir
        next_path = self.abs_path / name
        if next_path.exists() and not next_path.is_dir():
            raise ValueError(f"Expected '{next_path}' to be a directory, but it is a file.")
        new_child = GGDir(
            self.rel_path / name,
            parent=self,
            data_type_level=self.data_type_level,
            level=self.level + 1,
        )
        if new_child.exists():
            self.sub_dirs.append(new_child)
        if further_parts:
            return new_child.get_sub_dir(further_parts)
        return new_child

    def exists(self) -> bool:
        """Return ``True`` when this directory exists on disk."""
        if self.abs_path.exists():
            if not self.abs_path.is_dir():
                raise ValueError(f"Expected '{self.abs_path}' to be a directory, but it is a file.")
            return True
        return False

    def touch(self) -> Self:
        """Create the directory on disk if it does not exist."""
        if self.abs_path.exists() and not self.abs_path.is_dir():
            raise ValueError(f"Expected '{self.abs_path}' to be a directory, but it is a file.")
        self.abs_path.mkdir(parents=True, exist_ok=True)
        return self

    def all_files_paths(self, rec: bool = True) -> List[Path]:
        """Return file paths in this node and, optionally, its descendants.

        Args:
            rec: If ``True``, include files from all descendant nodes.
                If ``False``, only include files directly in this node.
        """
        file_paths = []
        for item in self.abs_path.iterdir():
            if item.is_file():
                file_paths.append(item)
        if rec:
            for sub_dir in self.filtered_sub_dirs:
                file_paths.extend(sub_dir.all_files_paths())
        return file_paths

    @property
    def filtered_sub_dirs(self) -> List[GGDir]:
        """Return direct child nodes that pass the active filters."""
        filters = self.root.filters.get(self.level + 1, None)
        if filters is None:
            return self.sub_dirs
        allowed = []
        for sub_dir in self.sub_dirs:
            if all(sub_dir.name != f[1:] for f in filters if f.startswith("!")) and any(
                sub_dir.name == f for f in filters if not f.startswith("!")
            ):
                allowed.append(sub_dir)
        return allowed

    @property
    def data_type_level_parent(self) -> GGDir:
        """Return the nearest ancestor (or self) marked as type separation level.

        Raises:
            GGDirNotFoundError: If no type separation level node can be found up to the root.
        """
        if self.is_data_type_level_parent:
            return self
        if self.parent is None:
            raise GGDirNotFoundError(f"No type separation level found in '{self.name}' and no parent to check.")
        return self.parent.data_type_level_parent

    @property
    def is_data_type_level_parent(self) -> bool:
        """Return ``True`` if this node's children are type branches (for example, ``images`` and ``labels``)."""
        return self.level == self.data_type_level - 1

    @property
    def data_type_dir(self) -> Optional[GGDir]:
        """Return the data type of this node"""
        if self.level == self.data_type_level:
            return self
        elif self.parent is not None:
            return self.parent.data_type_dir
        else:
            return None

    @property
    def data_type(self) -> Optional[str]:
        """Return the data type name of this node"""
        data_type_dir = self.data_type_dir
        if data_type_dir is not None:
            return data_type_dir.name
        return None

    def get_file(self, filename: str | Path) -> GGFile:
        """Return a file wrapper from this directory.

        Args:
            filename: Basename of the target file. Supports forward-slash separated paths.
        """
        if isinstance(filename, Path):
            filename = filename.as_posix()

        if "/" in filename:
            parts = filename.split("/")
            dir_parts = "/".join(parts[:-1])
            filename = parts[-1]
            t_dir = self.get_sub_dir(dir_parts)
            return t_dir.get_file(filename)
        file_path = self.abs_path / filename
        if file_path.exists():
            if file_path.is_file():
                return GGFile(self, filename)
            else:
                raise ValueError(f"Expected '{file_path}' to be a file, but it is a directory.")
        return GGFile(self, filename)

    def get_corresponding_file(
        self, cur_file: Path, data_type: str, extension: str, target_set: GGSet | None = None
    ) -> GGFile:
        """Map a file to its counterpart in another type separation branch.

        The relative path below the current type separation-level node is preserved while
        switching the first branch to ``data_type`` and replacing the file
        extension with ``extension``.

        Args:
            cur_file: Source file path.
            data_type: Target top-level type branch name.
            extension: New suffix, including leading dot.
            target_set: Optional GGSet to use as the root for the output path.
                When given, the output file is placed in this set instead of
                the source set, mirroring the same relative structure.
        """
        data_level = self.data_type_level_parent
        target_data_level = target_set.get_sub_dir(data_level.rel_path) if target_set is not None else data_level
        rel_path = cur_file.relative_to(data_level.abs_path)
        final_GGDir = target_data_level.get_sub_dir(data_type)

        for part in rel_path.parts[1:-1]:
            final_GGDir = final_GGDir.get_sub_dir(part)

        return final_GGDir.get_file(cur_file.with_suffix(extension).name)

    def get_corresponding_dir(self, cur_file: Path, data_type: str, target_set: GGSet | None = None) -> GGDir:
        """Map a file to its counterpart type branch.

        The relative path below the current type-level node is preserved while switching the first branch to ``data_type``.

        Args:
            cur_file: Source file path.
            data_type: Target top-level type branch name.
            target_set: Optional GGSet to use as the root for the output path.
                When given, the output directory is placed in this set instead
                of the source set, mirroring the same relative structure.
        """
        data_level = self.data_type_level_parent
        target_data_level = target_set.get_sub_dir(data_level.rel_path) if target_set is not None else data_level
        rel_path = cur_file.relative_to(data_level.abs_path)
        final_GGDir = target_data_level.get_sub_dir(data_type)

        for part in rel_path.parts[1:-1]:
            final_GGDir = final_GGDir.get_sub_dir(part)
        final_name = rel_path.parts[-1].rsplit(".", 1)[0]
        return final_GGDir.get_sub_dir(final_name)

    def get_corresponding_file_for_this_dir(
        self, data_type: str, extension: str, target_set: GGSet | None = None
    ) -> GGFile:
        """Map this directory to a file in another type branch with the same relative path.

        The relative path below the current type separation-level node is preserved while
        switching the first branch to ``data_type`` and replacing the directory name
        with a file of the same name and the specified extension.

        Args:
            data_type: Target top-level type branch name.
            extension: New suffix, including leading dot.
            target_set: Optional GGSet to use as the root for the output path.
                When given, the output file is placed in this set instead of
                the source set, mirroring the same relative structure.
        """
        data_level = self.data_type_level_parent
        target_data_level = target_set.get_sub_dir(data_level.rel_path) if target_set is not None else data_level
        rel_path = self.abs_path.relative_to(data_level.abs_path)
        final_GGDir = target_data_level.get_sub_dir(data_type)

        for part in rel_path.parts[1:]:
            final_GGDir = final_GGDir.get_sub_dir(part)

        filename = rel_path.parts[-1] + extension
        return final_GGDir.get_file(filename)

    @property
    def root(self) -> GGSet:
        """Return the root node of the current GGDir tree."""
        if self.parent is None:
            if not isinstance(self, GGSet):
                raise ValueError("Root node must be an instance of GGSet.")
            return self
        return self.parent.root

    def get_unique_sub_file_name(self, extension: str) -> str:
        """Generate a unique file name for a new file in this directory."""
        if not self.exists():
            return f"1{extension}"
        existing_files = {f.name for f in self.abs_path.iterdir() if f.is_file()}
        index = 1
        while True:
            candidate_name = f"{index}{extension}"
            if candidate_name not in existing_files:
                return candidate_name
            index += 1

    def get_new_sub_file(self, extension: str) -> GGFile:
        """Create a new file with a unique name in this directory and return its GGFile."""
        filename = self.get_unique_sub_file_name(extension)
        return GGFile(self, filename)

    def iterate(
        self,
        data_type: str | List[str] | Tuple[str, ...] | None = None,
        filter_endings: Iterable[str] = (),
        min_layer: int = 0,
    ) -> Generator[GGFile, None, None]:
        """Yield files for one logical data branch.

        Traversal behavior depends on node role:

        - Above the branch root: recurse until the branch root is reached.
        - At the branch root: select the direct child named ``data_type``.
        - Within the selected branch: yield files and recurse into descendants.

        Args:
            data_type: Optional: Data branch name(s) to iterate (e.g. ``"images"`` or ``["images", "labels"]``)
            filter_endings: Optional list of lowercase suffixes to include
                (e.g. ``".jpg"``, ``".txt"``). If empty, all files are
                yielded.
            min_layer: Optional minimum layer to start iteration from.

        Yields:
            ``GGFile`` objects matching traversal and filtering criteria.
        """
        if isinstance(data_type, list) or isinstance(data_type, tuple):
            for dt in data_type:
                yield from self.iterate(dt, filter_endings, min_layer)
            return

        if self.data_type_level > 0 and data_type is None and self.level > self.data_type_level + 1:
            for child in self.filtered_sub_dirs:
                yield from child.iterate(data_type, filter_endings, min_layer)
        elif self.data_type_level > 0 and data_type is not None and self.is_data_type_level_parent:
            if data_type is None:
                raise ValueError(f"Data branch name must be provided when iterating at data level '{self.name}'.")
            child = self.get_sub_dir(data_type)
            yield from child.iterate(data_type, filter_endings, min_layer)
        else:
            if self.level >= min_layer:
                for item in self.abs_path.iterdir():
                    if item.is_file():
                        if not filter_endings or item.suffix.lower() in filter_endings:
                            yield GGFile(self, item.name)
            for child in self.filtered_sub_dirs:
                yield from child.iterate(data_type, filter_endings, min_layer)

    def iterate_layer(self, layer: int) -> Generator[GGDir, None, None]:
        """Yield GGDir nodes at a specific layer"""
        if self.level == layer:
            yield self
        else:
            for child in self.filtered_sub_dirs:
                yield from child.iterate_layer(layer)

    def ancestor_at_level(self, target_level: int) -> GGDir:
        """Return the ancestor GGDir at a specific level."""
        if self.level == target_level:
            return self
        elif self.parent is None:
            raise GGDirNotFoundError(f"Reached root without finding target level {target_level}.")
        else:
            return self.parent.ancestor_at_level(target_level)

    def file_count(self, rec: bool = True, filter_endings: tuple[str, ...] | None = None) -> int:
        """Return the number of files in this node and, optionally, its descendants.

        Args:
            rec: If ``True``, include files from all descendant nodes.
                If ``False``, only include files directly in this node.
            filter_endings: Optional tuple of lowercase suffixes to include
                (e.g. ``(".jpg"``, ``".txt"``). If empty or ``None``, all files are
                counted.
        """
        count = 0
        for item in self.abs_path.iterdir():
            if item.is_file() and (not filter_endings or item.suffix.lower() in filter_endings):
                count += 1
        if rec:
            for sub_dir in self.filtered_sub_dirs:
                count += sub_dir.file_count(rec=True, filter_endings=filter_endings)
        return count

    def print_tree(self, indent: str = "", indent_steps: int = 2, filtered_out: bool = False) -> None:
        """Print the GGDir tree structure starting from this node."""
        ending_counts = {}
        for item in self.abs_path.iterdir():
            if item.is_file() and len(item.suffix) > 0:
                ending_counts[item.suffix.lower()] = ending_counts.get(item.suffix.lower(), 0) + 1
        if filtered_out:
            print(f"{indent}{self.name}/ (filtered out)")
        else:
            print(f"{indent}{self.name}/")
        for ending, count in ending_counts.items():
            print(f"{indent}  {ending}: {count}")
        if len(ending_counts) > 0:
            print()
        for child in self.sub_dirs:
            not_filtered = child.name in [c.name for c in self.filtered_sub_dirs]
            child.print_tree(indent + " " * indent_steps, indent_steps, filtered_out=not not_filtered)

    def print_counts(
        self,
        level: int,
        filter_endings: tuple[str, ...] | None = None,
        print_on_no_children: bool = True,
        indent: str = "",
        indent_steps: int = 2,
    ) -> None:
        """Print the file counts below this node at a specific layer, optionally filtered by file endings."""
        if level < self.level:
            raise ValueError(f"Target level {level} is above current node level {self.level}.")
        if self.level == level:
            count = self.file_count(rec=True, filter_endings=filter_endings)
            print(f"{indent}{self.name}/: {count} files")
        elif len(self.filtered_sub_dirs) > 0:
            print(f"{indent}{self.name}/")
            for child in self.filtered_sub_dirs:
                child.print_counts(
                    level, filter_endings, print_on_no_children, indent + " " * indent_steps, indent_steps
                )
        elif print_on_no_children:
            count = self.file_count(rec=True, filter_endings=filter_endings)
            print(f"{indent}{self.name}/: {count} files")

    def __str__(self) -> str:
        return f"GGDir at '{self.abs_path}'"

    def __repr__(self) -> str:
        return str(self.rel_path)

    def create_bulk_json_collection(
        self, name: str, layer: int, rel_paths: bool = False, bulk_files_root: GGSet | None = None
    ) -> GGBulkCollection:
        """Create a GGBulkCollection for writing or reading rows to JSON files across a layer.

        Args:
            name: Name of the bulk collection file (e.g., "bulk_data.json").
            layer: Layer at which to create the bulk collection.
            rel_paths: If True, store relative paths in the bulk collection; otherwise, store absolute paths.
            bulk_files_root: Optional GGSet to write annotation files into instead of the source set.
                Useful when the source dataset is read-only.
        """
        if rel_paths:
            key_strategy = RelativePathKeyMappingStrategy()
        else:
            key_strategy = DefaultKeyMappingStrategy()

        return GGBulkCollection(
            self,
            name,
            resolver_strategy=LayerResolver(layer),
            storage_strategy=JsonStorageStrategy(key_strategy),
            bulk_files_root=bulk_files_root,
        )

    def create_bulk_csv_collection(
        self,
        name: str,
        layer: int,
        rel_paths: bool = False,
        filename_col_name: str = "filename",
        caching: bool = False,
        bulk_files_root: GGSet | None = None,
    ) -> GGBulkCollection:
        """Create a GGBulkCollection for writing or reading rows to CSV files across a layer.

        Args:
            name: Name of the bulk collection file (e.g., "bulk_data.csv").
            layer: Layer at which to create the bulk collection.
            rel_paths: If True, store relative paths in the bulk collection; otherwise, store absolute paths.
            filename_col_name: Name of the column that stores file names in the CSV.
            caching: If True, enable caching for the bulk collection. Warning: when caching=True, and you call write with an file that already exists in the bulk collection, the new data will overwrite the old data. When caching=False, the new data will be appended to the bulk collection, and you will have to handle duplicates yourself.
            bulk_files_root: Optional GGSet to write annotation files into instead of the source set.
                Useful when the source dataset is read-only.
        """
        if rel_paths:
            key_strategy = RelativePathKeyMappingStrategy()
        else:
            key_strategy = DefaultKeyMappingStrategy()

        if caching:
            storage_strategy = CsvCachingStorageStrategy(key_strategy, filename_col_name)
        else:
            storage_strategy = CsvStorageStrategy(key_strategy, filename_col_name)

        return GGBulkCollection(
            self,
            name,
            resolver_strategy=LayerResolver(layer),
            storage_strategy=storage_strategy,
            bulk_files_root=bulk_files_root,
        )


class GGSet(GGDir):
    """Root GGDir specialization used as the main user entry point."""

    filters: Dict[int, Tuple[str, ...]] = {}

    def __init__(self, path: str | Path, data_type_level: int = -1) -> None:
        self.root_path = Path(path).resolve()
        super().__init__(Path(), parent=None, data_type_level=data_type_level, level=0)

    def add_filter_allow_only(self, level: int, *allowed_dirs) -> None:
        """Add a filter to only allow certain subdirectories at a given level.

        Args:
            level: Depth from the root node where the filter should be applied.
            allowed_dirs: Iterable of directory names to allow at the specified level.
        """
        self.filters[level] = tuple(allowed_dirs)

    def add_filter_exclude(self, level: int, *excluded_dirs) -> None:
        """Add a filter to exclude certain subdirectories at a given level.

        Args:
            level: Depth from the root node where the filter should be applied.
            excluded_dirs: Iterable of directory names to exclude at the specified level.
        """
        self.filters[level] = tuple([f"!{dir_name}" for dir_name in excluded_dirs])


TRUEISH_VALUES = {"true", "1", "yes", "y"}


class GGFile:
    """Wrapper around a concrete dataset file with typed read helpers."""

    def __init__(self, GGDir: GGDir, file_name: str) -> None:
        """Initialize a file wrapper.

        Args:
            GGDir: Owning GGDir node.
            file_name: File name relative to ``GGDir.abs_path``.
        """
        self.ggdir = GGDir
        self.file_name = file_name

    @property
    def abs_path(self) -> Path:
        """Return the absolute path to this file."""
        return self.ggdir.abs_path / self.file_name

    @property
    def rel_path(self) -> Path:
        """Return the path from the GGDir root to this file."""
        return self.ggdir.rel_path / self.file_name

    @property
    def data_type(self) -> Optional[str]:
        """Return the data type of this file, derived from its GGDir."""
        return self.ggdir.data_type

    def exists(self) -> bool:
        """Return ``True`` when this file exists on disk."""
        if self.abs_path.exists():
            if not self.abs_path.is_file():
                raise ValueError(f"Expected '{self.abs_path}' to be a file, but it is a directory.")
            return True
        return False

    def touch(self) -> Self:
        """Create the file on disk if it does not exist."""
        self.abs_path.parent.mkdir(parents=True, exist_ok=True)
        if self.abs_path.exists() and not self.abs_path.is_file():
            raise ValueError(f"Expected '{self.abs_path}' to be a file, but it is a directory.")
        self.abs_path.touch(exist_ok=True)
        return self

    def get_corresponding_file(self, data_type: str, extension: str, target_set: GGSet | None = None) -> GGFile:
        """Resolve this file's counterpart in another data branch.

        Args:
            data_type: Target top-level type branch name.
            extension: New suffix, including leading dot.
            target_set: Optional GGSet to resolve the output file in.
        """
        return self.ggdir.get_corresponding_file(self.abs_path, data_type, extension, target_set=target_set)

    def get_corresponding_dir(self, data_type: str, target_set: GGSet | None = None) -> GGDir:
        """Resolve this file's dataset branch counterpart.

        Args:
            data_type: Target top-level type branch name.
            target_set: Optional GGSet to resolve the output directory in.
        """
        return self.ggdir.get_corresponding_dir(self.abs_path, data_type, target_set=target_set)

    def get_corresponding_file_in_same_dir(self, extension: str, target_set: GGSet | None = None) -> GGFile:
        """Resolve this file's counterpart in the same directory but with a different extension.

        Args:
            extension: New suffix, including leading dot.
            target_set: Optional GGSet to resolve the output file in.
        """
        target_dir = self.ggdir if target_set is None else target_set.get_sub_dir(self.ggdir.rel_path)
        return target_dir.get_file(self.abs_path.with_suffix(extension).name)

    def get_corresponding_dir_in_same_dir(self, target_set: GGSet | None = None) -> GGDir:
        """Resolve this file's counterpart directory in the same directory with the same name as the file (without extension).

        Args:
            target_set: Optional GGSet to resolve the output directory in.
        """
        target_dir = self.ggdir if target_set is None else target_set.get_sub_dir(self.ggdir.rel_path)
        return target_dir.get_sub_dir(self.abs_path.with_suffix("").name)

    def read_image(self) -> Any:
        """Read an image file with OpenCV.

        Supported suffixes: ``.jpg``, ``.jpeg``, ``.png``, ``.bmp``.

        Raises:
            ValueError: If the file extension is not a supported image type.
        """
        if self.abs_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            raise ValueError(f"File '{self.abs_path}' is not a supported image format.")
        return cv2.imread(str(self.abs_path))

    def read_text(self) -> str:
        """Read the file as plain text.

        Supported suffixes: ``.txt``, ``.csv``, ``.json``, ``.yaml``, ``.yml``.

        Raises:
            ValueError: If the file extension is not a supported text type.
        """
        if self.abs_path.suffix.lower() not in [".txt", ".csv", ".json", ".yaml", ".yml"]:
            raise ValueError(f"File '{self.abs_path}' is not a supported text format.")
        return self.abs_path.read_text()

    def read_single_bool(self) -> bool:
        """Parse the file as a single boolean value.

        True-ish values are defined by ``TRUEISH_VALUES``.
        """
        return self.read_text().strip().lower() in TRUEISH_VALUES

    def read_single_int(self) -> int:
        """Parse the file as a single integer value."""
        return int(self.read_text().strip())

    def read_single_float(self) -> float:
        """Parse the file as a single floating-point value."""
        return float(self.read_text().strip())

    def read_int_list(self, sep: str = " ") -> List[int]:
        """Parse a separator-delimited list of integers from text."""
        return [int(x) for x in self.read_text().strip().split(sep)]

    def read_float_list(self, sep: str = " ") -> List[float]:
        """Parse a separator-delimited list of floats from text."""
        return [float(x) for x in self.read_text().strip().split(sep)]

    def read_bool_list(self, sep: str = " ") -> List[bool]:
        """Parse a separator-delimited list of booleans from text."""
        return [x.strip().lower() in TRUEISH_VALUES for x in self.read_text().strip().split(sep)]

    def read_json(self) -> Dict[str, Any]:
        """Parse the file content as JSON."""
        return json.loads(self.read_text())

    def read_yaml(self) -> Dict[str, Any]:
        """Parse the file content as YAML using ``yaml.safe_load``."""
        return yaml.safe_load(self.read_text())

    def read_dataframe(self) -> pd.DataFrame:
        """Read the file as a CSV into a pandas DataFrame."""
        return pd.read_csv(self.abs_path)

    def read_np_array(self) -> Any:
        """Read the file as a NumPy array using np.load."""
        return np.load(self.abs_path)

    def write_text(self, content: str) -> None:
        """Write text content to the file."""
        self.touch()
        self.abs_path.write_text(content)

    def write_image(self, image: Any) -> None:
        """Write an image to the file using OpenCV."""
        self.touch()
        cv2.imwrite(str(self.abs_path), image)

    def write_json(self, data: Dict[str, Any]) -> None:
        """Write a dictionary to the file as JSON."""
        self.touch()
        self.write_text(json.dumps(data, indent=4))

    def write_yaml(self, data: Dict[str, Any]) -> None:
        """Write a dictionary to the file as YAML."""
        self.touch()
        self.write_text(yaml.dump(data))

    def write_dataframe(self, df: pd.DataFrame) -> None:
        """Write a pandas DataFrame to the file as CSV."""
        self.touch()
        df.to_csv(self.abs_path, index=False)

    def write_np_array(self, array: Any) -> None:
        """Write a NumPy array to the file using np.save."""
        self.touch()
        np.save(self.abs_path, array)

    def __str__(self) -> str:
        return f"GGFile at '{self.rel_path}'" + ("*" if not self.abs_path.exists() else "")

    def __repr__(self) -> str:
        return str(self.rel_path) + (" *" if not self.abs_path.exists() else "")


class BulkFileResolver(ABC):
    """Abstract base class for resolving bulk file instances."""

    @abstractmethod
    def resolve(self, file: GGFile, bulk_collection: "GGBulkCollection") -> GGFile:
        """Resolve and return a bulk file instance for a given GGDir."""
        pass

    @abstractmethod
    def all_files(self, bulk_collection: "GGBulkCollection") -> List[GGFile]:
        """Return a list of all GGFiles in the bulk collection."""
        pass


class LayerResolver(BulkFileResolver):
    def __init__(self, layer: int):
        self.layer = layer

    def resolve(self, ref_file: GGFile, bulk_collection: "GGBulkCollection") -> GGFile:
        d = ref_file.ggdir.ancestor_at_level(self.layer - 1).rel_path
        target_dir = bulk_collection.bulk_files_root.get_sub_dir(d)
        return target_dir.get_file(bulk_collection.file_name)

    def all_files(self, bulk_collection: "GGBulkCollection") -> List[GGFile]:
        files = []
        for dir in bulk_collection.bulk_files_root.iterate_layer(self.layer - 1):
            f = dir.get_file(bulk_collection.file_name)
            if f.exists():
                files.append(f)
        return files


class KeyMappingStrategy(ABC):
    """Abstract base class for defining key mapping"""

    @abstractmethod
    def to_store_key(self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> str:
        """Return a unique key for a given reference file."""
        pass

    @abstractmethod
    def from_store_key(self, key: str, context_file: GGFile, bulk_collection: "GGBulkCollection") -> str:
        """Return a GGFile corresponding to a given key."""
        pass


class DefaultKeyMappingStrategy(KeyMappingStrategy):
    """Default key mapping strategy that uses the relative path of the reference file."""

    def to_store_key(self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> str:
        return str(ref_file.rel_path)

    def from_store_key(self, key: str, context_file: GGFile, bulk_collection: "GGBulkCollection") -> str:
        return key


class RelativePathKeyMappingStrategy(KeyMappingStrategy):
    """Key mapping strategy that uses the relative path of the reference file from the context file's directory."""

    def to_store_key(self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> str:
        return str(ref_file.rel_path.relative_to(bulk_file.ggdir.rel_path))

    def from_store_key(self, key: str, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> str:
        return (bulk_file.ggdir.rel_path / key).as_posix()


class BulkStorageStrategy(ABC):
    """Abstract base class for defining storage strategies for bulk collections."""

    @abstractmethod
    def write(self, ref_file: GGFile, bulk_file: GGFile, data: Any, bulk_collection: "GGBulkCollection") -> None:
        """Write data for a given reference file."""
        pass

    @abstractmethod
    def read_dataframe(self, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> pd.DataFrame:
        """Read data for a given bulk file and return it as a pandas DataFrame."""
        pass

    @abstractmethod
    def read_dict(self, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> Dict[str, Any]:
        """Read data for a given bulk file and return it as a dictionary."""
        pass

    @abstractmethod
    def existing_files_set(self, bulk_file: GGFile, bulk_collection: "GGBulkCollection") -> set[str]:
        """Return a set of existing reference file keys in the bulk file."""
        pass

    @abstractmethod
    def read_for_file(
        self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: "GGBulkCollection"
    ) -> Optional[Dict[str, Any]]:
        """Read data for a specific reference file from the corresponding bulk file."""
        pass

    @abstractmethod
    def flush(self, bulk_collection: "GGBulkCollection") -> None:
        """Flush any buffered data to the bulk file."""
        pass

    def iterate(
        self, bulk_file: GGFile, bulk_collection: "GGBulkCollection"
    ) -> Generator[Tuple[GGFile, Dict[str, Any]], None, None]:
        """Iterate over all key-value pairs in the bulk file."""
        data_dict = self.read_dict(bulk_file, bulk_collection)
        for file_path, value in data_dict.items():
            ref_file = bulk_collection.data_root.get_file(file_path)
            if ref_file.exists():
                yield ref_file, value


class GGBulkCollection:
    """Class representing a bulk collection of data across multiple files."""

    def __init__(
        self,
        data_root: GGDir,
        file_name: str,
        resolver_strategy: BulkFileResolver,
        storage_strategy: BulkStorageStrategy,
        bulk_files_root: GGDir | None = None,
    ) -> None:
        self.data_root = data_root
        self.file_name = file_name
        self.resolver_strategy = resolver_strategy
        self.storage_strategy = storage_strategy
        self.bulk_files_root = bulk_files_root if bulk_files_root is not None else data_root

    def write(self, ref_file: GGFile, data: Any) -> None:
        """Write data for a given reference file."""
        bulk_file = self.resolver_strategy.resolve(ref_file, self)
        self.storage_strategy.write(ref_file, bulk_file, data, self)

    def read_dataframe(self) -> pd.DataFrame:
        """Read data for the entire bulk collection and return it as a pandas DataFrame."""
        all_dataframes = []
        for bulk_file in self.resolver_strategy.all_files(self):
            df = self.storage_strategy.read_dataframe(bulk_file, self)
            all_dataframes.append(df)
        if len(all_dataframes) == 0:
            return pd.DataFrame()
        return pd.concat(all_dataframes, ignore_index=True)

    def read_dict(self) -> Dict[str, Any]:
        """Read data for the entire bulk collection and return it as a dictionary."""
        all_data = {}
        for bulk_file in self.resolver_strategy.all_files(self):
            data = self.storage_strategy.read_dict(bulk_file, self)
            all_data.update(data)
        return all_data

    def existing_files_set(self) -> set[str]:
        """Return a set of existing reference file keys in the entire bulk collection."""
        all_keys = set()
        for bulk_file in self.resolver_strategy.all_files(self):
            keys = self.storage_strategy.existing_files_set(bulk_file, self)
            all_keys.update(keys)
        return all_keys

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        """Read data for a specific reference file from the corresponding bulk file."""
        bulk_file = self.resolver_strategy.resolve(ref_file, self)
        return self.storage_strategy.read_for_file(ref_file, bulk_file, self)

    def flush(self) -> None:
        """Flush any buffered data to the bulk files."""
        self.storage_strategy.flush(self)

    def __enter__(self) -> "GGBulkCollection":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.flush()

    def iterate(self) -> Generator[Tuple[GGFile, Dict[str, Any]], None, None]:
        """Iterate over all key-value pairs in the bulk collection."""
        for bulk_file in self.resolver_strategy.all_files(self):
            yield from self.storage_strategy.iterate(bulk_file, self)

    def __iter__(self) -> Generator[Tuple[GGFile, Dict[str, Any]], None, None]:
        return self.iterate()


class JsonStorageStrategy(BulkStorageStrategy):
    """Storage strategy for bulk collections using JSON files."""

    def __init__(self, key_resolver: KeyMappingStrategy):
        self.key_resolver = key_resolver
        self.buffer: Dict[str, Dict[str, Any]] = {}

    def _load(self, bulk_file: GGFile) -> Dict[str, Any]:
        if not bulk_file.exists():
            return {}
        return bulk_file.read_json()

    def _get_buffer(self, bulk_file: GGFile) -> Dict[str, Any]:
        key = str(bulk_file.rel_path)
        if key not in self.buffer:
            self.buffer[key] = self._load(bulk_file)
        return self.buffer[key]

    def read_for_file(
        self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: GGBulkCollection
    ) -> Dict[str, Any] | None:
        bulk_data = self._get_buffer(bulk_file)
        if not bulk_data:
            return None
        store_key = self.key_resolver.to_store_key(ref_file, bulk_file, bulk_collection)
        return bulk_data.get(store_key)

    def write(self, ref_file: GGFile, bulk_file: GGFile, data: Any, bulk_collection: GGBulkCollection) -> None:
        bulk_data = self._get_buffer(bulk_file)
        store_key = self.key_resolver.to_store_key(ref_file, bulk_file, bulk_collection)
        if store_key in bulk_data:
            bulk_data[store_key].update(data)
        else:
            bulk_data[store_key] = data

    def read_dict(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> Dict[str, Any]:
        d = self._get_buffer(bulk_file)
        return {self.key_resolver.from_store_key(k, bulk_file, bulk_collection): v for k, v in d.items()}

    def read_dataframe(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> pd.DataFrame:
        d = self.read_dict(bulk_file, bulk_collection)
        l = [{"filename": k, **v} for k, v in d.items()]
        return pd.DataFrame(l)

    def existing_files_set(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> set[str]:
        bulk_data = self._get_buffer(bulk_file)
        return set(bulk_data.keys())

    def flush(self, bulk_collection: GGBulkCollection) -> None:
        for bulk_file_key, data in self.buffer.items():
            if len(data) > 0:
                bulk_file = bulk_collection.bulk_files_root.get_file(bulk_file_key)
                bulk_file.write_json(data)
        self.buffer.clear()


class CsvBuffer:
    """Buffer for CSV storage strategy to hold rows in memory before writing to disk."""

    def __init__(self, file: GGFile, filename_col_name: str):
        self.file = file
        self.filename_col_name = filename_col_name
        self.rows: List[List[str]] = []
        self.headers: List[str] | None = None
        self.should_write_headers: bool = True
        if self.file.exists():
            with open(self.file.abs_path, "r") as f:
                first_line = f.readline()
                if first_line:
                    self.headers = first_line.strip().split(",")
                    if self.headers[0] != self.filename_col_name:
                        raise ValueError(
                            f"CSV file '{self.file.abs_path}' has a different first column name '{self.headers[0]}' than expected '{self.filename_col_name}'."
                        )
                    if len(self.headers) > 1:
                        for line in f:
                            self.rows.append(line.strip().split(","))
                        self.should_write_headers = False

    def write(self, store_key: str, data: Dict[str, Any]) -> None:
        if self.headers is None:
            self.headers = [self.filename_col_name] + list(data.keys())
        row = [store_key] + [str(data.get(h, "")) for h in self.headers[1:]]
        self.rows.append(row)

    def flush(self) -> None:
        if len(self.rows) == 0:
            return
        if not self.file.exists():
            self.file.touch()
        with open(self.file.abs_path, "a") as f:
            writer = csv.writer(f)
            if self.should_write_headers:
                if self.headers is None:
                    raise ValueError("Headers are not set, cannot write to CSV.")
                writer.writerow(self.headers)
            for row in self.rows:
                writer.writerow(row)
        self.rows.clear()
        self.should_write_headers = False


class CsvStorageStrategy(BulkStorageStrategy):
    """Storage strategy for bulk collections using CSV files. This strategy does not hold a buffer in memory, and writes directly to disk."""

    def __init__(self, key_resolver: KeyMappingStrategy, filename_col_name: str = "filename"):
        self.key_resolver = key_resolver
        self.filename_col_name = filename_col_name
        self.buffers: Dict[str, CsvBuffer] = {}

    def _get_csv_buffer(self, bulk_file: GGFile) -> CsvBuffer:
        key = str(bulk_file.rel_path)
        if key not in self.buffers:
            self.buffers[key] = CsvBuffer(bulk_file, self.filename_col_name)
        return self.buffers[key]

    def write(self, ref_file: GGFile, bulk_file: GGFile, data: Any, bulk_collection: GGBulkCollection) -> None:
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data)}.")
        buffer = self._get_csv_buffer(bulk_file)
        store_key = self.key_resolver.to_store_key(ref_file, bulk_file, bulk_collection)
        buffer.write(store_key, data)

    def read_dataframe(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> pd.DataFrame:
        if not bulk_file.exists():
            return pd.DataFrame(columns=[self.filename_col_name])
        df = pd.read_csv(bulk_file.abs_path)
        df[self.filename_col_name] = df[self.filename_col_name].map(
            lambda x: self.key_resolver.from_store_key(str(x), bulk_file, bulk_collection)
        )
        return df

    def read_dict(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> Dict[str, Any]:
        df = self.read_dataframe(bulk_file, bulk_collection)
        return df.set_index(self.filename_col_name).to_dict(orient="index")  # type: ignore

    def existing_files_set(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> set[str]:
        existing_files = set()
        if bulk_file.exists():
            with open(bulk_file.abs_path, "r") as f:
                first_line = f.readline()
                if first_line:
                    headers = first_line.strip().split(",")
                    if headers[0] != self.filename_col_name:
                        raise ValueError(
                            f"CSV file '{bulk_file.abs_path}' has a different first column name '{headers[0]}' than expected '{self.filename_col_name}'."
                        )
                    for line in f:
                        key = line.strip().split(",")[0].strip()
                        existing_files.add(self.key_resolver.from_store_key(key, bulk_file, bulk_collection))
        return existing_files

    def read_for_file(
        self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: GGBulkCollection
    ) -> Optional[Dict[str, Any]]:
        if not bulk_file.exists():
            return None
        store_key = self.key_resolver.to_store_key(ref_file, bulk_file, bulk_collection)
        with open(bulk_file.abs_path, "r") as f:
            first_line = f.readline()
            if not first_line:
                return None
            headers = first_line.strip().split(",")
            if headers[0] != self.filename_col_name:
                raise ValueError(
                    f"CSV file '{bulk_file.abs_path}' has a different first column name '{headers[0]}' than expected '{self.filename_col_name}'."
                )
            for line in f:
                row = line.strip().split(",")
                if row[0].strip() == store_key:
                    return {headers[i]: row[i] for i in range(1, len(headers))}
        return None

    def flush(self, bulk_collection: GGBulkCollection) -> None:
        for buffer in self.buffers.values():
            buffer.flush()
        self.buffers.clear()


class CsvCachingStorageStrategy(CsvStorageStrategy):
    """Storage strategy for bulk collections using CSV files with caching enabled. This strategy holds a buffer in memory, and writes to disk only when flushed."""

    def __init__(self, key_resolver: KeyMappingStrategy, filename_col_name: str = "filename"):
        super().__init__(key_resolver, filename_col_name)
        self.dfs: Dict[str, pd.DataFrame] = {}

    def _get_dataframe(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> pd.DataFrame:
        key = str(bulk_file.rel_path)
        if key not in self.dfs:
            if bulk_file.exists():
                self.dfs[key] = pd.read_csv(bulk_file.abs_path)
            else:
                self.dfs[key] = pd.DataFrame(columns=[self.filename_col_name])
        return self.dfs[key]

    def write(self, ref_file: GGFile, bulk_file: GGFile, data: Any, bulk_collection: GGBulkCollection) -> None:
        if not isinstance(data, dict):
            raise ValueError(f"Data must be a dictionary, got {type(data)}.")
        df = self._get_dataframe(bulk_file, bulk_collection)
        store_key = self.key_resolver.to_store_key(ref_file, bulk_file, bulk_collection)
        new_row = {self.filename_col_name: store_key, **data}
        if store_key in df[self.filename_col_name].values:
            # df.loc[df[self.filename_col_name] == store_key, list(data.keys())] = list(data.values())
            filt = df[self.filename_col_name] == store_key
            for key, value in data.items():
                df.loc[filt, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self.dfs[str(bulk_file.rel_path)] = df

    def flush(self, bulk_collection: GGBulkCollection) -> None:
        for key, df in self.dfs.items():
            if not df.empty:
                bulk_file = bulk_collection.bulk_files_root.get_file(key)
                bulk_file.touch()
                df.to_csv(bulk_file.abs_path, index=False)
        self.dfs.clear()

    def read_dataframe(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> pd.DataFrame:
        r = self._get_dataframe(bulk_file, bulk_collection).copy()
        r[self.filename_col_name] = r[self.filename_col_name].map(
            lambda x: self.key_resolver.from_store_key(str(x), bulk_file, bulk_collection)
        )
        return r

    def read_dict(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> Dict[str, Any]:
        df = self.read_dataframe(bulk_file, bulk_collection)
        return df.set_index(self.filename_col_name).to_dict(orient="index")  # type: ignore

    def read_for_file(
        self, ref_file: GGFile, bulk_file: GGFile, bulk_collection: GGBulkCollection
    ) -> Dict[str, Any] | None:
        df = self._get_dataframe(bulk_file, bulk_collection)
        store_key = self.key_resolver.to_store_key(ref_file, bulk_file, bulk_collection)
        row = df[df[self.filename_col_name] == store_key]
        if row.empty:
            return None
        d = row.iloc[0].to_dict()
        d.pop(self.filename_col_name, None)
        return d  # type: ignore

    def existing_files_set(self, bulk_file: GGFile, bulk_collection: GGBulkCollection) -> set[str]:
        df = self._get_dataframe(bulk_file, bulk_collection)
        return set(
            df[self.filename_col_name].map(
                lambda x: self.key_resolver.from_store_key(str(x), bulk_file, bulk_collection)
            )
        )
