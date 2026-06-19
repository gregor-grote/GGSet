"""Utilities for navigating hierarchical dataset folders.

This module provides two classes:

- ``GGDir`` to model a directory tree and iterate through files by logical
    dataset categories.
- ``GGFile`` to represent a concrete file and offer typed convenience
    readers (image, text, numbers, JSON, YAML, CSV).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Generator, Tuple, Union, Literal, overload
import os
from pathlib import Path
import cv2
import json
import yaml
import pandas as pd
from abc import ABC, abstractmethod

__all__ = [
    "GGDir",
    "GGFile",
    "GGSet",
    "GGNotFoundError",
    "GGDirNotFoundError",
    "GGFileNotFoundError",
    "GGBulkBase",
    "GGBulkCsvBase",
    "GGBulkCsvFileCollection",
    "GGBulkJsonBase",
    "GGBulkJsonFileCollection",
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
        name: str = "root",
        type_sep_level: int = -1,
        level: int = 0,
    ) -> None:
        """Initialize a GGDir node and eagerly build its children.

        Args:
            path: Directory represented by this node.
            parent: Parent node, or ``None`` for the root node.
            name: Human-readable node name. If not root, it must match the directory name.
            type_sep_level: Remaining depth until type-separation nodes are
                reached (``0`` means this node is type-separation level, ``-1`` means no type-separation).
            level: Depth from the root node.
        """
        assert level != 0 or isinstance(
            self, GGSet
        ), "Only the root node can have level 0, and it must be an instance of GGSet."
        self.path = Path(path)
        self.parent = parent
        self.sub_dirs: List[GGDir] = []
        self.name = name
        self.type_sep_level = type_sep_level
        self.level = level
        self._build()

    def _build(self) -> None:
        """Populate ``sub_dirs`` with subdirectory nodes."""
        for item in self.path.iterdir():
            if item.is_dir():
                child = GGDir(
                    item, parent=self, name=item.name, type_sep_level=self.type_sep_level, level=self.level + 1
                )
                self.sub_dirs.append(child)

    def get_sub_dir(self, name: str, force_create: bool = False) -> GGDir:
        """Return a direct child node by name.

        Args:
            name: Child directory name.
            force_create: If True, create the child node if it does not exist.
        Raises:
            GGDirNotFoundError: If no direct child with ``name`` exists and ``force_create`` is False.
        """
        for sub_dir in self.sub_dirs:
            if sub_dir.name == name:
                return sub_dir
        if force_create:
            next_path = self.path / name
            next_path.mkdir(exist_ok=True)
            new_child = GGDir(
                next_path,
                parent=self,
                name=name,
                type_sep_level=self.type_sep_level,
                level=self.level + 1,
            )
            self.sub_dirs.append(new_child)
            return new_child
        raise GGDirNotFoundError(
            f"Child with name '{name}' not found in '{self.name}', available children: {[sub_dir.name for sub_dir in self.sub_dirs]}"
        )

    def all_files_paths(self, rec: bool = True) -> List[Path]:
        """Return file paths in this node and, optionally, its descendants.

        Args:
            rec: If ``True``, include files from all descendant nodes.
                If ``False``, only include files directly in this node.
        """
        file_paths = []
        for item in self.path.iterdir():
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
    def type_sep_level_parent(self) -> GGDir:
        """Return the nearest ancestor (or self) marked as type separation level.

        Raises:
            GGDirNotFoundError: If no type separation level node can be found up to the root.
        """
        if self.is_type_sep_level_parent:
            return self
        if self.parent is None:
            raise GGDirNotFoundError(f"No type separation level found in '{self.name}' and no parent to check.")
        return self.parent.type_sep_level_parent

    @property
    def is_type_sep_level_parent(self) -> bool:
        """Return ``True`` if this node's children are type branches (for example, ``images`` and ``labels``)."""
        return self.level == self.type_sep_level - 1

    @overload
    def get_file(self, filename: str, force_create: Literal[True]) -> GGFile: ...

    @overload
    def get_file(self, filename: str, force_create: Literal[False] = False) -> GGFile | None: ...

    def get_file(self, filename: str, force_create: bool = False) -> GGFile | None:
        """Return a file from this directory, if it exists.

        Args:
            filename: Basename of the target file.
            force_create: If ``True``, create the file if it does not exist.

        Returns:
            ``GGFile`` if found. If ``force_create`` is ``True``, a ``GGFile``
            is always returned.
        """
        file_path = self.path / filename
        if file_path.exists():
            if file_path.is_file():
                return GGFile(self, filename)
            else:
                raise ValueError(f"Expected '{file_path}' to be a file, but it is a directory.")
        elif force_create:
            file_path.touch()
            return GGFile(self, filename)
        return None

    @overload
    def get_corresponding_file(
        self, cur_file: Path, target_type: str, target_extension: str, force_create: Literal[True]
    ) -> GGFile: ...

    @overload
    def get_corresponding_file(
        self, cur_file: Path, target_type: str, target_extension: str, force_create: Literal[False] = False
    ) -> GGFile | None: ...

    def get_corresponding_file(
        self, cur_file: Path, target_type: str, target_extension: str, force_create: bool = False
    ) -> GGFile | None:
        """Map a file to its counterpart in another type separation branch.

        The relative path below the current type separation-level node is preserved while
        switching the first branch to ``target_type`` and replacing the file
        extension with ``target_extension``.

        Args:
            cur_file: Source file path.
            target_type: Target top-level type branch name.
            target_extension: New suffix, including leading dot.
            force_create: If ``True``, create an empty file at the target path if it does not exist.
        Returns:
            Matching ``GGFile`` in the target branch if available.
            If ``force_create`` is ``True``, a ``GGFile`` is always returned.
        """
        data_level = self.type_sep_level_parent
        rel_path = cur_file.relative_to(data_level.path)
        try:
            final_GGDir = data_level.get_sub_dir(target_type, force_create=force_create)
        except GGDirNotFoundError:
            return None

        for part in rel_path.parts[1:-1]:
            try:
                final_GGDir = final_GGDir.get_sub_dir(part, force_create=force_create)
            except GGDirNotFoundError:
                return None

        try:
            return final_GGDir.get_file(cur_file.with_suffix(target_extension).name, force_create=force_create)
        except GGFileNotFoundError:
            return None

    @overload
    def get_corresponding_dir(self, cur_file: Path, target_type: str, force_create: Literal[True]) -> GGDir: ...

    @overload
    def get_corresponding_dir(
        self, cur_file: Path, target_type: str, force_create: Literal[False] = False
    ) -> GGDir | None: ...

    def get_corresponding_dir(self, cur_file: Path, target_type: str, force_create: bool = False) -> GGDir | None:
        """Map a file to its counterpart type branch.

        The relative path below the current type-level node is preserved while switching the first branch to ``target_type``.

        Args:
            cur_file: Source file path.
            target_type: Target top-level type branch name.
            force_create: If ``True``, create missing directories along the target path.
        Returns:
            Matching ``GGDir`` type branch if available.
            If ``force_create`` is ``True``, a ``GGDir`` is always returned.
        """

        data_level = self.type_sep_level_parent
        rel_path = cur_file.relative_to(data_level.path)
        try:
            final_GGDir = data_level.get_sub_dir(target_type, force_create=force_create)
        except GGDirNotFoundError:
            return None

        for part in rel_path.parts[1:-1]:
            try:
                final_GGDir = final_GGDir.get_sub_dir(part, force_create=force_create)
            except GGDirNotFoundError:
                return None
        final_name = rel_path.parts[-1].rsplit(".", 1)[0]

        try:
            return final_GGDir.get_sub_dir(final_name, force_create=force_create)
        except GGDirNotFoundError:
            return None

    @property
    def root(self) -> GGSet:
        """Return the root node of the current GGDir tree."""
        if self.parent is None:
            if not isinstance(self, GGSet):
                raise ValueError("Root node must be an instance of GGSet.")
            return self
        return self.parent.root

    @property
    def rel_path(self) -> Path:
        """Return the path from the root to this node."""
        if self.parent is None:
            return Path()
        return self.path.relative_to(self.root.path)

    def get_unique_child_file_name(self, extension: str) -> str:
        """Generate a unique file name for a new file in this directory."""
        existing_files = {f.name for f in self.path.iterdir() if f.is_file()}
        index = 1
        while True:
            candidate_name = f"{index}{extension}"
            if candidate_name not in existing_files:
                return candidate_name
            index += 1

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
        if data_type is None and self.level > self.type_sep_level + 1:
            for child in self.filtered_sub_dirs:
                yield from child.iterate(data_type, filter_endings, min_layer)
        elif data_type is not None and self.is_type_sep_level_parent:
            if data_type is None:
                raise ValueError(f"Data branch name must be provided when iterating over data level '{self.name}'.")
            child = self.get_sub_dir(data_type)
            yield from child.iterate(data_type, filter_endings, min_layer)
        else:
            if self.level >= min_layer:
                for item in self.path.iterdir():
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

    def write_text_file(self, content: str, name: str | None = None, extension: str = ".txt") -> GGFile:
        """Write text content to a new file with a unique name in this directory."""
        filename = name or self.get_unique_child_file_name(extension)
        file_path = self.path / filename
        file_path.write_text(content)
        return GGFile(self, filename)

    def print_tree(self, indent: str = "", indent_steps: int = 2, filtered_out: bool = False) -> None:
        """Print the GGDir tree structure starting from this node."""
        ending_counts = {}
        for item in self.path.iterdir():
            if item.is_file():
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

    def __str__(self) -> str:
        return f"GGDir at '{self.path}'"

    def __repr__(self) -> str:
        return str(self.rel_path)


class GGSet(GGDir):
    """Root GGDir specialization used as the main user entry point."""

    filters: Dict[int, Tuple[str, ...]] = {}

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

    def crate_bulk_csv_writer(
        self, name: str, layer: int, cols: List[str], save_rel_paths: bool = False
    ) -> GGBulkCsvFileCollection:
        """Create a GGBulkCsvFileCollection for writing rows to CSV files across a layer."""
        return GGBulkCsvFileCollection(self, name, layer, cols, save_rel_paths=save_rel_paths)

    def crate_bulk_json_writer(
        self, name: str, layer: int, cols: List[str] | None = None, save_rel_paths: bool = False
    ) -> GGBulkJsonFileCollection:
        """Create a GGBulkJsonFileCollection for writing rows to JSON files across a layer."""
        del cols
        return GGBulkJsonFileCollection(self, name, layer, save_rel_paths=save_rel_paths)


TRUEISH_VALUES = {"true", "1", "yes", "y"}


class GGFile:
    """Wrapper around a concrete dataset file with typed read helpers."""

    def __init__(self, GGDir: GGDir, file_name: str) -> None:
        """Initialize a file wrapper.

        Args:
            GGDir: Owning GGDir node.
            file_name: File name relative to ``GGDir.path``.
        """
        self.ggdir = GGDir
        self.file_name = file_name

    @property
    def abs_path(self) -> Path:
        """Return the absolute path to this file."""
        return self.ggdir.path / self.file_name

    @property
    def rel_path(self) -> Path:
        """Return the path from the GGDir root to this file."""
        return self.abs_path.relative_to(self.ggdir.root.path)

    @overload
    def get_corresponding_file(
        self, target_type: str, target_extension: str, force_create: Literal[True]
    ) -> GGFile: ...

    @overload
    def get_corresponding_file(
        self, target_type: str, target_extension: str, force_create: Literal[False] = False
    ) -> GGFile | None: ...

    def get_corresponding_file(
        self, target_type: str, target_extension: str, force_create: bool = False
    ) -> GGFile | None:
        """Resolve this file's counterpart in another data branch."""
        return self.ggdir.get_corresponding_file(
            self.abs_path, target_type, target_extension, force_create=force_create
        )

    @overload
    def get_corresponding_dir(self, target_type: str, force_create: Literal[True]) -> GGDir: ...

    @overload
    def get_corresponding_dir(self, target_type: str, force_create: Literal[False] = False) -> GGDir | None: ...

    def get_corresponding_dir(self, target_type: str, force_create: bool = False) -> GGDir | None:
        """Resolve this file's dataset branch counterpart."""
        return self.ggdir.get_corresponding_dir(self.abs_path, target_type, force_create=force_create)

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

    def write_annotation_text(self, target_type: str, content: Any, file_extension: str = ".txt") -> None:
        """Write text content to a file in the target data branch with the same relative path."""
        target_file = self.get_corresponding_file(target_type, file_extension, force_create=True)
        if target_file is None:
            raise ValueError(
                f"Failed to create or get target file for writing annotation text in data '{target_type}'."
            )
        target_file.abs_path.write_text(str(content))

    def __str__(self) -> str:
        return f"GGFile at '{self.rel_path}'"

    def __repr__(self) -> str:
        return str(self.rel_path)


class GGBulkBase(ABC):
    def __init__(self, save_rel_paths: bool = False) -> None:
        self.save_rel_paths = save_rel_paths

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


class GGBulkCsvBase(GGBulkBase, ABC):
    """CSV branch of bulk writers."""

    def __init__(self, name: str, cols: List[str], save_rel_paths: bool = False) -> None:
        super().__init__(save_rel_paths=save_rel_paths)
        self.name = name
        self.cols = cols
        if not self.cols:
            self.cols = ["Filename"]
        if self.cols[0] != "Filename":
            self.cols.insert(0, "Filename")

    def _store_filename(self, ref_file: GGFile, bulk_dir: GGDir) -> str:
        if self.save_rel_paths:
            return str(ref_file.abs_path.relative_to(bulk_dir.path))
        return str(ref_file.rel_path)

    def _normalize_filename(self, filename: str, bulk_dir: GGDir) -> str:
        if self.save_rel_paths:
            return str((bulk_dir.rel_path / Path(filename)).as_posix())
        return filename

    def write_dict_row(self, ref_file: GGFile, data: Dict[str, Any]) -> None:
        values = [data.get(col, "") for col in self.cols[1:]]
        self.write(ref_file, values)

    def read_dict(self) -> Dict[str, Dict[str, Any]]:
        df = self.read_dataframe()
        result = {}
        for _, row in df.iterrows():
            filename = row["Filename"]
            result[filename] = {col: row[col] for col in self.cols[1:]}
        return result

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        df = self.read_dataframe()
        filename = str(ref_file.rel_path)
        row = df[df["Filename"] == filename]
        if row.empty:
            return None
        return row.iloc[0].to_dict()  # type: ignore

    def read_row_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        return self.read_for_file(ref_file)


class GGBulkCsvFileCollection(GGBulkCsvBase):
    def __init__(self, ggset: GGSet, name: str, layer: int, cols: List[str], save_rel_paths: bool = False) -> None:
        super().__init__(name, cols, save_rel_paths=save_rel_paths)
        self.ggset = ggset
        self.layer = layer
        self.files: Dict[str, GGBulkCsvSingleFile] = {}

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, list):
            raise TypeError("CSV bulk writers expect list-like row data. Use write_dict_row for dictionary input.")
        target_dir = ref_file.ggdir.ancestor_at_level(self.layer - 1)
        filename = str(target_dir.rel_path)
        if filename not in self.files:
            self.files[filename] = GGBulkCsvSingleFile(
                target_dir, f"{self.name}.csv", self.cols, save_rel_paths=self.save_rel_paths
            )
        self.files[filename].write(ref_file, data)

    def read_dataframe(self) -> pd.DataFrame:
        self.flush()
        dfs = []
        for cur_dir in self.ggset.iterate_layer(self.layer - 1):
            df_file = cur_dir.get_file(f"{self.name}.csv")
            if df_file is not None:
                dfs.append(
                    GGBulkCsvSingleFile(
                        cur_dir, f"{self.name}.csv", self.cols, save_rel_paths=self.save_rel_paths
                    ).read_dataframe()
                )
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=self.cols)

    def flush(self) -> None:
        for file in self.files.values():
            file.flush()


class GGBulkCsvSingleFile(GGFile, GGBulkCsvBase):
    def __init__(self, ggdir: GGDir, file_name: str, cols: List[str], save_rel_paths: bool = False) -> None:
        if file_name.endswith(".csv"):
            name = file_name[:-4]
        else:
            name = file_name
            file_name += ".csv"

        GGFile.__init__(self, ggdir, file_name)
        GGBulkCsvBase.__init__(self, name, cols, save_rel_paths=save_rel_paths)

        try:
            self.df = pd.read_csv(self.abs_path)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=self.cols)
        if len(self.df.columns) == 0:
            self.df = pd.DataFrame(columns=self.cols)
        if self.df.columns.tolist() != self.cols:
            raise ValueError(
                f"Existing CSV file '{self.abs_path}' has columns {self.df.columns.tolist()} which do not match expected columns {self.cols}."
            )

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, list):
            raise TypeError("CSV bulk writers expect list-like row data. Use write_dict_row for dictionary input.")
        filename = self._store_filename(ref_file, self.ggdir)
        new_row = {"Filename": filename}
        for col, value in zip(self.cols[1:], data):
            new_row[col] = value
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def read_dataframe(self) -> pd.DataFrame:
        df = self.df.copy()
        if not df.empty:
            df["Filename"] = df["Filename"].map(lambda filename: self._normalize_filename(str(filename), self.ggdir))
        return df

    def flush(self) -> None:
        self.df.to_csv(self.abs_path, index=False)


class GGBulkJsonBase(GGBulkBase, ABC):
    """JSON branch of bulk writers."""

    def __init__(self, name: str, save_rel_paths: bool = False) -> None:
        super().__init__(save_rel_paths=save_rel_paths)
        self.name = name

    def _store_filename(self, ref_file: GGFile, bulk_dir: GGDir) -> str:
        if self.save_rel_paths:
            return str(ref_file.abs_path.relative_to(bulk_dir.path))
        return str(ref_file.rel_path)

    def _normalize_filename(self, filename: str, bulk_dir: GGDir) -> str:
        if self.save_rel_paths:
            return str((bulk_dir.rel_path / Path(filename)).as_posix())
        return filename

    def write_dict_row(self, ref_file: GGFile, data: Dict[str, Any]) -> None:
        self.write(ref_file, data)

    def read_dict(self) -> Dict[str, Dict[str, Any]]:
        df = self.read_dataframe()
        result = {}
        for _, row in df.iterrows():
            filename = row["Filename"]
            result[filename] = {col: row[col] for col in df.columns if col != "Filename"}
        return result

    def read_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        return self.read_dict().get(str(ref_file.rel_path))

    def read_row_for_file(self, ref_file: GGFile) -> Optional[Dict[str, Any]]:
        return self.read_for_file(ref_file)


class GGBulkJsonFileCollection(GGBulkJsonBase):
    def __init__(self, ggset: GGSet, name: str, layer: int, save_rel_paths: bool = False) -> None:
        super().__init__(name, save_rel_paths=save_rel_paths)
        self.ggset = ggset
        self.layer = layer
        self.files: Dict[str, GGBulkJsonSingleFile] = {}

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("JSON bulk writers expect dictionary row data. Use write_dict_row for schema-shaped input.")
        target_dir = ref_file.ggdir.ancestor_at_level(self.layer - 1)
        filename = str(target_dir.rel_path)
        if filename not in self.files:
            self.files[filename] = GGBulkJsonSingleFile(
                target_dir, f"{self.name}.json", save_rel_paths=self.save_rel_paths
            )
        self.files[filename].write(ref_file, data)

    def read_dataframe(self) -> pd.DataFrame:
        self.flush()
        dfs = []
        for cur_dir in self.ggset.iterate_layer(self.layer - 1):
            df_file = cur_dir.get_file(f"{self.name}.json")
            if df_file is not None:
                dfs.append(
                    GGBulkJsonSingleFile(
                        cur_dir, f"{self.name}.json", save_rel_paths=self.save_rel_paths
                    ).read_dataframe()
                )
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame(columns=["Filename"])

    def flush(self) -> None:
        for file in self.files.values():
            file.flush()


class GGBulkJsonSingleFile(GGFile, GGBulkJsonBase):
    def __init__(self, ggdir: GGDir, file_name: str, save_rel_paths: bool = False) -> None:
        if file_name.endswith(".json"):
            name = file_name[:-5]
        else:
            name = file_name
            file_name += ".json"

        GGFile.__init__(self, ggdir, file_name)
        GGBulkJsonBase.__init__(self, name, save_rel_paths=save_rel_paths)

        if self.abs_path.exists():
            loaded = self.read_json()
            if not isinstance(loaded, dict):
                raise ValueError(f"Existing JSON file '{self.abs_path}' must contain a dictionary at the root.")
            self.rows: Dict[str, Dict[str, Any]] = {}
            for key, value in loaded.items():
                if not isinstance(key, str) or not isinstance(value, dict):
                    raise ValueError(f"Existing JSON file '{self.abs_path}' must map filename keys to object rows.")
                self.rows[key] = value
        else:
            self.rows = {}

    def write(self, ref_file: GGFile, data: Any) -> None:
        if not isinstance(data, dict):
            raise TypeError("JSON bulk writers expect dictionary row data. Use write_dict_row for schema-shaped input.")
        filename = self._store_filename(ref_file, self.ggdir)
        self.rows[filename] = data

    def read_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"Filename": self._normalize_filename(filename, self.ggdir), **row} for filename, row in self.rows.items()]
        )

    def flush(self) -> None:
        self.abs_path.write_text(json.dumps(self.rows, indent=2, sort_keys=True))
