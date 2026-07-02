"""Utilities for navigating hierarchical dataset folders.

This module provides two classes:

- ``GGDir`` to model a directory tree and iterate through files by logical
    dataset categories.
- ``GGFile`` to represent a concrete file and offer typed convenience
    readers (image, text, numbers, JSON, YAML, CSV).
"""

from __future__ import annotations

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
    Literal,
    overload,
    TypeVar,
    Generic,
)
import os
from pathlib import Path
import cv2
import json
import yaml
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
    "GGBulkBase",
    "GGBulkJsonFileCollection",
    "GGBulkCsvFileCollection",
]


class GGNotFoundError(ValueError):
    """Raised when a requested directory or file is not found in the GGDir tree."""


class GGDirNotFoundError(GGNotFoundError):
    """Raised when a requested directory is not found in the GGDir tree."""


class GGFileNotFoundError(GGNotFoundError):
    """Raised when a requested file is not found in the GGDir tree."""


class GGDir:

    def __init__(
        self,
        path: str | Path,
        parent: GGDir | None = None,
        data_type_level: int = -1,
        level: int = 0,
    ) -> None:
        """Initialize a GGDir node and eagerly build its children.

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
        self.sub_dirs: List[GGDir] = []
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

    def _build(self) -> None:
        """Populate ``sub_dirs`` with subdirectory nodes."""
        if not self.abs_path.exists():
            return
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
                self.sub_dirs.append(child)

    @staticmethod
    def _navigate_rel_path(start: GGDir, rel_path: Path) -> GGDir:
        """Walk *start* down the parts of *rel_path* using get_sub_dir.

        Parts that are empty or ``'.'`` are skipped so that navigating a
        ``Path('.')`` (i.e. the root) simply returns *start* unchanged.

        Args:
            start: The GGDir to begin navigation from.
            rel_path: Relative path whose parts are traversed one by one.
        """
        current = start
        for part in rel_path.parts:
            if part in ("", "."):
                continue
            current = current.get_sub_dir(part)
        return current

    def get_sub_dir(self, name: str) -> GGDir:
        """Return a direct child node by name.

        Args:
            name: Child directory name.
        """
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
        return self.abs_path.exists() and self.abs_path.is_dir()

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
    def data_type(self) -> Optional[GGDir]:
        """Return the data type of this node"""
        if self.level == self.data_type_level:
            return self
        elif self.parent is not None:
            return self.parent.data_type
        else:
            return None

    def get_file(self, filename: str) -> GGFile:
        """Return a file wrapper from this directory.

        Args:
            filename: Basename of the target file.
        """
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
        target_data_level = (
            GGDir._navigate_rel_path(target_set, data_level.rel_path) if target_set is not None else data_level
        )
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
        target_data_level = (
            GGDir._navigate_rel_path(target_set, data_level.rel_path) if target_set is not None else data_level
        )
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
        target_data_level = (
            GGDir._navigate_rel_path(target_set, data_level.rel_path) if target_set is not None else data_level
        )
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
        # file_path = self.abs_path / filename
        # file_path.touch()
        return GGFile(self, filename)

    def iterate(
        self, data_type: str | None = None, filter_endings: Iterable[str] = (), min_layer: int = 0
    ) -> Generator[GGFile, None, None]:
        """Yield files for one logical data branch.

        Traversal behavior depends on node role:

        - Above the branch root: recurse until the branch root is reached.
        - At the branch root: select the direct child named ``data_type``.
        - Within the selected branch: yield files and recurse into descendants.

        Args:
            data_type: Optional: Data branch name to iterate (e.g. ``"images"``)
            filter_endings: Optional list of lowercase suffixes to include
                (e.g. ``".jpg"``, ``".txt"``). If empty, all files are
                yielded.
            min_layer: Optional minimum layer to start iteration from.

        Yields:
            ``GGFile`` objects matching traversal and filtering criteria.
        """
        if self.data_type_level > 0 and data_type is None and self.level > self.data_type_level + 1:
            for child in self.filtered_sub_dirs:
                yield from child.iterate(data_type, filter_endings, min_layer)
        elif self.data_type_level > 0 and data_type is not None and self.is_data_type_level_parent:
            if data_type is None:
                raise ValueError(f"Data branch name must be provided when iterating over data level '{self.name}'.")
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
        self, name: str, layer: int, rel_paths: bool = False, write_root: GGSet | None = None
    ) -> GGBulkJsonFileCollection:
        """Create a GGBulkJsonFileCollection for writing or reading rows to JSON files across a layer.

        Args:
            name: Name of the bulk collection file (e.g., "bulk_data.json").
            layer: Layer at which to create the bulk collection.
            rel_paths: If True, store relative paths in the bulk collection; otherwise, store absolute paths.
            write_root: Optional GGSet to write annotation files into instead of the source set.
                Useful when the source dataset is read-only.
        """
        return GGBulkJsonFileCollection(self, name, layer, rel_paths=rel_paths, write_root=write_root)

    def create_bulk_csv_collection(
        self,
        name: str,
        layer: int,
        rel_paths: bool = False,
        filename_col_name: str = "filename",
        caching: bool = False,
        write_root: GGSet | None = None,
    ) -> GGBulkCsvFileCollection:
        """Create a GGBulkCsvFileCollection for writing or reading rows to CSV files across a layer.

        Args:
            name: Name of the bulk collection file (e.g., "bulk_data.csv").
            layer: Layer at which to create the bulk collection.
            rel_paths: If True, store relative paths in the bulk collection; otherwise, store absolute paths.
            filename_col_name: Name of the column that stores file names in the CSV.
            caching: If True, enable caching for the bulk collection. Warning: when caching=True, and you call write with an file that already exists in the bulk collection, the new data will overwrite the old data. When caching=False, the new data will be appended to the bulk collection, and you will have to handle duplicates yourself.
            write_root: Optional GGSet to write annotation files into instead of the source set.
                Useful when the source dataset is read-only.
        """
        return GGBulkCsvFileCollection(
            self,
            name,
            layer,
            rel_paths=rel_paths,
            filename_col_name=filename_col_name,
            caching=caching,
            write_root=write_root,
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
        target_dir = self.ggdir if target_set is None else GGDir._navigate_rel_path(target_set, self.ggdir.rel_path)
        return target_dir.get_file(self.abs_path.with_suffix(extension).name)

    def get_corresponding_dir_in_same_dir(self, target_set: GGSet | None = None) -> GGDir:
        """Resolve this file's counterpart directory in the same directory with the same name as the file (without extension).

        Args:
            target_set: Optional GGSet to resolve the output directory in.
        """
        target_dir = self.ggdir if target_set is None else GGDir._navigate_rel_path(target_set, self.ggdir.rel_path)
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


class GGBulkBase(ABC):
    @abstractmethod
    def write(self, ref_file: GGFile, data: Any) -> None:
        pass

    @abstractmethod
    def read_dataframe(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def read_dict(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_existing_files_set(self) -> set[str]:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.flush()

    @abstractmethod
    def iter(self) -> Iterator[Tuple[GGFile, Dict[str, Any]]]:
        pass

    def __iter__(self) -> Iterator[Tuple[GGFile, Dict[str, Any]]]:
        return self.iter()


T = TypeVar("T", bound="GGBulkSingleFileBase")


class GGBulkCollectionBase(GGBulkBase, ABC, Generic[T]):
    """Base class for bulk writers that manage multiple files across a GGSet."""

    def __init__(
        self,
        file_name: str,
        layer: int,
        data_root: GGDir,
        rel_paths: bool = False,
        write_root: GGDir | None = None,
    ) -> None:
        if layer < 1:
            raise ValueError("Layer must be >= 1.")
        self.file_name = file_name
        self.layer = layer
        self.data_root = data_root
        self.write_root = write_root if write_root is not None else data_root
        self.rel_paths = rel_paths
        self.files: Dict[str, T] = {}

    def _get_corresponding_write_dir(self, source_dir: GGDir) -> GGDir:
        """Return the directory in *write_root* that mirrors *source_dir* in *data_root*.

        When ``write_root`` and ``data_root`` are the same object, *source_dir*
        is returned unchanged.  Otherwise the relative path of *source_dir* is
        navigated from *write_root*.
        """
        if self.write_root is self.data_root:
            return source_dir
        return GGDir._navigate_rel_path(self.write_root, source_dir.rel_path)

    def _iter_write_dirs(self) -> Generator[GGDir, None, None]:
        """Yield write-root directories at the bulk layer.

        When ``write_root`` equals ``data_root``, iterates ``write_root``
        directly.  Otherwise derives write dirs by mapping each ``data_root``
        dir through ``_get_corresponding_write_dir``, which keeps ``write_root``
        in sync even when it was built before its subdirectories were created.
        """
        if self.write_root is self.data_root:
            yield from self.write_root.iterate_layer(self.layer - 1)
        else:
            for dir in self.data_root.iterate_layer(self.layer - 1):
                yield self._get_corresponding_write_dir(dir)

    @abstractmethod
    def _create_bulk_file(self, ggdir: GGDir) -> T:
        """Create a new bulk file instance for a given GGDir."""
        pass

    @overload
    def get_bulk_file(self, ggdir: GGDir, create: Literal[True]) -> T: ...

    @overload
    def get_bulk_file(self, ggdir: GGDir, create: Literal[False]) -> Optional[T]: ...

    def get_bulk_file(self, ggdir: GGDir, create: bool = False) -> Optional[T]:
        """Get or create the bulk file for a specific GGDir."""
        bulk_dir = ggdir.ancestor_at_level(self.layer - 1)
        cache_key = str(bulk_dir.rel_path)
        if cache_key not in self.files:
            target_file = bulk_dir.abs_path / self.file_name
            if target_file.exists() or create:
                self.files[cache_key] = self._create_bulk_file(bulk_dir)
            else:
                return None

        return self.files[cache_key]

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        """Read data for a specific reference file from the corresponding bulk file."""
        source_dir = ref_file.ggdir.ancestor_at_level(self.layer - 1)
        write_dir = self._get_corresponding_write_dir(source_dir)
        bulk_file = self.get_bulk_file(write_dir, create=False)
        if bulk_file is None:
            return None
        return bulk_file.read_for_file(ref_file)

    def read_dataframe(self) -> pd.DataFrame:
        """Read and concatenate data from all bulk files into a single DataFrame."""
        dfs = []
        for dir in self._iter_write_dirs():
            bulk_file = self.get_bulk_file(dir, create=False)
            if bulk_file is not None:
                dfs.append(bulk_file.read_dataframe())
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def read_dict(self) -> Dict[str, Any]:
        """Read and merge data from all bulk files into a single dictionary."""
        result = {}
        for dir in self._iter_write_dirs():
            bulk_file = self.get_bulk_file(dir, create=False)
            if bulk_file is not None:
                result.update(bulk_file.read_dict())
        return result

    def get_existing_files_set(self) -> Set[str]:
        """Get a set of all existing reference file names across all bulk files."""
        existing_files = set()
        for dir in self._iter_write_dirs():
            bulk_file = self.get_bulk_file(dir, create=False)
            if bulk_file is not None:
                existing_files.update(bulk_file.get_existing_files_set())
        return existing_files

    def get_existing_annotation_files_set(self) -> Set[T]:
        """Get a set of all existing annotation files across all bulk files."""
        existing_files = set()
        for dir in self._iter_write_dirs():
            f = self.get_bulk_file(dir, create=False)
            if f is not None:
                existing_files.add(f)
        return existing_files

    def flush(self) -> None:
        for bulk_file in self.files.values():
            bulk_file.flush()

    def write(self, ref_file: GGFile, data: Any) -> None:
        """Write data for a specific reference file to the corresponding bulk file."""
        source_dir = ref_file.ggdir.ancestor_at_level(self.layer - 1)
        write_dir = self._get_corresponding_write_dir(source_dir)
        bulk_file = self.get_bulk_file(write_dir, create=True)
        bulk_file.write(ref_file, data)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for bulk_file in self.files.values():
            bulk_file.__exit__(exc_type, exc_val, exc_tb)
        self.files.clear()

    def iter(self) -> Iterator[Tuple[GGFile, Dict[str, Any]]]:
        """Iterate over all bulk files managed by this collection."""
        for dir in self._iter_write_dirs():
            bulk_file = self.get_bulk_file(dir, create=False)
            if bulk_file is not None:
                data = bulk_file.read_dict()
                for filename, row_data in data.items():
                    file = self.data_root.root.get_file(filename)
                    if file.exists():
                        yield file, row_data
                    else:
                        raise GGFileNotFoundError(f"File '{filename}' not found in GGSet.")
            self.__exit__(None, None, None)


class GGBulkSingleFileBase(GGBulkBase, GGFile, ABC):
    """Base class for bulk writers that manage a single file within a GGDir."""

    def __init__(self, ggdir: GGDir, parent: GGBulkCollectionBase) -> None:
        self.ggdir = ggdir
        self.parent = parent
        GGFile.__init__(self, ggdir, parent.file_name)

    def _store_filename(self, ref_file: GGFile) -> str:
        if self.parent.rel_paths:
            return str(ref_file.abs_path.relative_to(self.ggdir.abs_path))
        return str(ref_file.rel_path)

    def _normalize_filename(self, filename: str) -> str:
        if self.parent.rel_paths:
            return str((self.ggdir.rel_path / Path(filename)).as_posix())
        return filename

    def iter(self) -> Iterator[Tuple[GGFile, Dict[str, Any]]]:
        """Iterate over all rows in the bulk file, yielding reference files and their associated data."""
        data = self.read_dict()
        for filename, row_data in data.items():
            file = self.parent.data_root.root.get_file(filename)
            if file.exists():
                yield file, row_data
            else:
                raise GGFileNotFoundError(f"File '{filename}' not found in GGSet.")


class GGBulkCsvFileCollection(GGBulkCollectionBase["GGBulkCsvSingleFile"]):
    def __init__(
        self,
        root: GGDir,
        file_name: str,
        layer: int,
        rel_paths: bool = False,
        filename_col_name: str = "filename",
        caching: bool = False,
        write_root: GGSet | None = None,
    ) -> None:
        if not file_name.endswith(".csv"):
            file_name += ".csv"
        super().__init__(file_name=file_name, layer=layer, data_root=root, rel_paths=rel_paths, write_root=write_root)
        self.filename_col_name = filename_col_name
        self.caching = caching

    def _create_bulk_file(self, ggdir: GGDir) -> "GGBulkCsvSingleFile | GGBulkCachingCsvFileCollection":
        if self.caching:
            return GGBulkCachingCsvFileCollection(ggdir, self)
        else:
            return GGBulkCsvSingleFile(ggdir, self)


class GGBulkCsvSingleFile(GGBulkSingleFileBase):
    def __init__(
        self,
        ggdir: GGDir,
        parent: GGBulkCsvFileCollection,
    ) -> None:
        super().__init__(ggdir, parent)
        self.parent = parent
        self.handler = None
        self.cols = self._read_column_names()

    def _read_column_names(self) -> Optional[List[str]]:
        if self.abs_path.exists() and self.abs_path.is_file():
            with self.abs_path.open() as f:
                header = f.readline().strip().split(",")
                if len(header) == 0:
                    return None
                if header[0] != self.parent.filename_col_name:
                    raise ValueError(
                        f"CSV file '{self.abs_path}' does not have '{self.parent.filename_col_name}' as the first column, cannot read columns."
                    )
                return header
        return None

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("Dynamic CSV bulk writers expect dictionary row data.")
        filename = self._store_filename(ref_file)
        if self.cols is None:
            self.cols = [self.parent.filename_col_name] + list(data.keys())
            self.abs_path.parent.mkdir(parents=True, exist_ok=True)
            with self.abs_path.open("w") as f:
                f.write(",".join(self.cols) + "\n")
        else:
            new_cols = [col for col in data.keys() if col not in self.cols]
            if new_cols:
                raise ValueError(
                    f"New columns {new_cols} are not in the existing CSV columns {self.cols}. Dynamic CSV does not support adding new columns after initialization."
                )
        row = [filename] + [str(data.get(col, "")) for col in self.cols[1:]]
        if self.handler is None:
            self.handler = self.abs_path.open("a")
        self.handler.write(",".join(row) + "\n")

    def read_dataframe(self) -> pd.DataFrame:
        self.flush()
        df = pd.read_csv(self.abs_path)
        df[self.parent.filename_col_name] = df[self.parent.filename_col_name].apply(
            lambda x: self._normalize_filename(x)
        )
        return df

    def read_dict(self) -> Dict[str, Any]:
        df = self.read_dataframe()
        result = {}
        for _, row in df.iterrows():
            filename = row[self.parent.filename_col_name]
            result[filename] = row.drop(labels=[self.parent.filename_col_name]).to_dict()
        return result

    def flush(self) -> None:
        if self.handler is not None:
            self.handler.flush()

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        if not ref_file.abs_path.is_relative_to(self.ggdir.abs_path):
            raise GGFileNotFoundError(f"File '{ref_file.rel_path}' is not in bulk branch '{self.ggdir.rel_path}'.")
        self.flush()
        filename = self._store_filename(ref_file)
        with self.abs_path.open() as f:
            header = f.readline().strip().split(",")
            if len(header) == 0:
                return None
            if header[0] != self.parent.filename_col_name:
                raise ValueError(
                    f"CSV file '{self.abs_path}' does not have '{self.parent.filename_col_name}' as the first column, cannot use lookup."
                )
            for line in f:
                row = line.strip().split(",")
                if row[0] == filename:
                    return {col: value for col, value in zip(header[1:], row[1:])}
        return None

    def get_existing_files_set(self) -> set[str]:
        self.flush()
        existing_files = set()
        with self.abs_path.open() as f:
            header = f.readline().strip().split(",")
            if len(header) == 0:
                return existing_files
            if header[0] != self.parent.filename_col_name:
                raise ValueError(
                    f"CSV file '{self.abs_path}' does not have '{self.parent.filename_col_name}' as the first column, cannot determine existing files set."
                )
            for line in f:
                row = line.strip().split(",")
                existing_files.add(self._normalize_filename(row[0]))
        return existing_files

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler is not None:
            self.handler.flush()
            self.handler.close()
            self.handler = None


class GGBulkCachingCsvFileCollection(GGBulkSingleFileBase):
    def __init__(
        self,
        ggdir: GGDir,
        parent: GGBulkCsvFileCollection,
    ) -> None:
        super().__init__(ggdir, parent)
        self.parent = parent
        self.df = None

    def _get_df(self) -> pd.DataFrame | None:
        if self.df is None:
            if self.abs_path.exists():
                if not self.abs_path.is_file():
                    raise ValueError(f"Expected a file at '{self.abs_path}', but found a directory.")
                df = pd.read_csv(self.abs_path)
                if not df.empty:
                    self.df = df
        return self.df

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("Dynamic CSV bulk writers expect dictionary row data.")
        filename = self._store_filename(ref_file)
        df = self._get_df()
        if df is None:
            df = pd.DataFrame(columns=[self.parent.filename_col_name] + list(data.keys()))
        if filename in df[self.parent.filename_col_name].values:
            df.loc[df[self.parent.filename_col_name] == filename, list(data.keys())] = pd.Series(data)
        else:
            new_row = {self.parent.filename_col_name: filename}
            new_row.update(data)
            self.df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    def read_dataframe(self) -> pd.DataFrame:
        df = self._get_df()
        if df is None:
            return pd.DataFrame(columns=[self.parent.filename_col_name])
        df = df.copy()
        df[self.parent.filename_col_name] = df[self.parent.filename_col_name].apply(
            lambda x: self._normalize_filename(x)
        )
        return df

    def read_dict(self) -> Dict[str, Any]:
        df = self.read_dataframe()
        result = {}
        for _, row in df.iterrows():
            filename = row[self.parent.filename_col_name]
            result[filename] = row.drop(labels=[self.parent.filename_col_name]).to_dict()
        return result

    def flush(self) -> None:
        if self.df is not None:
            self.df.to_csv(self.abs_path, index=False)

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        df = self._get_df()
        if df is None:
            return None
        filename = self._store_filename(ref_file)
        row = df[df[self.parent.filename_col_name] == filename]
        if not row.empty:
            return row.iloc[0].drop(labels=[self.parent.filename_col_name]).to_dict()  # type: ignore
        return None

    def get_existing_files_set(self) -> set[str]:
        df = self._get_df()
        if df is None:
            return set()
        return {self._normalize_filename(filename) for filename in df[self.parent.filename_col_name].tolist()}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class GGBulkJsonFileCollection(GGBulkCollectionBase["GGBulkJsonSingleFile"]):
    def __init__(
        self, root: GGDir, file_name: str, layer: int, rel_paths: bool = False, write_root: GGSet | None = None
    ) -> None:
        if not file_name.endswith(".json"):
            file_name += ".json"
        super().__init__(file_name=file_name, layer=layer, data_root=root, rel_paths=rel_paths, write_root=write_root)

    def _create_bulk_file(self, ggdir: GGDir) -> "GGBulkJsonSingleFile":
        return GGBulkJsonSingleFile(ggdir, self)


class GGBulkJsonSingleFile(GGBulkSingleFileBase):
    def __init__(
        self,
        ggdir: GGDir,
        parent: GGBulkJsonFileCollection,
    ):
        super().__init__(ggdir, parent)
        self.parent = parent
        self.data = self._read_json_file()

    def _read_json_file(self) -> Dict[str, Dict[str, Any]]:
        if self.abs_path.exists():
            loaded = self.read_json()
            if not isinstance(loaded, dict):
                raise ValueError(f"Existing JSON file '{self.abs_path}' must contain a dictionary at the root.")
            data = {}
            for key, value in loaded.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    raise ValueError(f"Existing JSON file '{self.abs_path}' must map filename keys to object rows.")
                data[key] = value
        else:
            data = {}

        return data

    def _update_data(self) -> None:
        disk_state = self._read_json_file()
        for filename, row_data in self.data.items():
            if filename in disk_state:
                disk_state[filename].update(row_data)
            else:
                disk_state[filename] = row_data
        self.data = disk_state

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("JSON bulk writers expect dictionary row data. Use write_list_row for list input.")
        filename = self._store_filename(ref_file)
        if filename in self.data:
            self.data[filename].update(data)
        else:
            self.data[filename] = data

    def read_dataframe(self) -> pd.DataFrame:
        self._update_data()
        rows = []
        for filename, row_data in self.data.items():
            row = {"filename": self._normalize_filename(filename)}
            row.update(row_data)
            rows.append(row)
        return pd.DataFrame(rows)

    def read_dict(self) -> Dict[str, Any]:
        self._update_data()
        result = {}
        for filename, row_data in self.data.items():
            normalized_filename = self._normalize_filename(filename)
            result[normalized_filename] = row_data
        return result

    def flush(self) -> None:
        self._update_data()
        self.write_json(self.data)

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        self._update_data()
        filename = self._store_filename(ref_file)
        if filename in self.data:
            return self.data[filename]
        return None

    def get_existing_files_set(self) -> set[str]:
        self._update_data()
        return {self._normalize_filename(filename) for filename in self.data.keys()}
