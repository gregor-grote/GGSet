# GGSet

GGSet is a small utility library for working with hierarchical dataset folders.
It builds a directory tree, lets you iterate files by logical data branch, resolves
corresponding files across branches, and provides bulk CSV/JSON sidecar writers for
dataset metadata.

## What It Solves

Many datasets are organized as parallel folder trees such as:

```text
dataset/
	train/
		images/
			sample1.png
		labels/
			sample1.txt
	test/
		images/
			sample2.png
		labels/
			sample2.txt
```

GGSet helps you:

- traverse one branch such as `images`
- find the matching file in another branch such as `labels`
- create missing corresponding files or directories
- read typed file content through a lightweight `GGFile` wrapper
- store metadata per directory layer in CSV or JSON files

## Installation

```bash
pip install https://github.com/gregor-grote/GGSet
```


## Core Concepts

### `GGSet`

`GGSet` is the root object. It stores the absolute dataset root path and builds a tree
of `GGDir` nodes below it.

### `GGDir`

`GGDir` represents a directory in the dataset tree.

- `rel_path` is the path from the dataset root
- `abs_path` is computed from `GGSet.root_path / rel_path`
- `name` is derived from the path rather than stored separately

### `GGFile`

`GGFile` wraps a concrete file and provides:

- `abs_path` and `rel_path`
- typed readers such as `read_text`, `read_json`, `read_yaml`, `read_dataframe`
- simple scalar helpers such as `read_single_int`, `read_single_float`, `read_bool_list`
- corresponding-file resolution across dataset branches

### `type_sep_level`

`type_sep_level` tells GGSet where the parallel data branches live.

Example:

```text
dataset/
	train/
		images/
		labels/
```

Here the branch names are `images` and `labels`, which are at depth `2` from the root,
so `type_sep_level=2`.

## Quick Start

```python
from ggset import GGSet

ggset = GGSet("dataset", type_sep_level=2)

for image_file in ggset.iterate("images"):
    label_file = image_file.get_corresponding_file("labels", ".txt")
    print(image_file.rel_path)
    if label_file is not None:
        print(label_file.read_text())
```

## Common Usage

### Iterate one branch

```python
from ggset import GGSet

ggset = GGSet("dataset", type_sep_level=1)

for data_file in ggset.iterate("data"):
    print(data_file.rel_path, data_file.read_text())
```

### Iterate all files

```python
for file in ggset.iterate():
    print(file.rel_path)
```

### Resolve corresponding files

```python
image_file = ggset.get_sub_dir("train").get_sub_dir("images").get_file("sample1.png")
assert image_file is not None

label_file = image_file.get_corresponding_file("labels", ".txt")
if label_file is not None:
    print(label_file.read_text())
```

### Create missing corresponding files

```python
image_file = ggset.get_sub_dir("data").get_file("sample1.png")
assert image_file is not None

label_file = image_file.get_corresponding_file("labels", ".txt", force_create=True)
label_file.write_text("annotation")
```

### Create nested output directories

```python
source_file = ggset.get_sub_dir("data").get_file("sample1.txt")
assert source_file is not None

annotation_dir = source_file.get_corresponding_dir("labels", force_create=True)
new_file = annotation_dir.get_file("0.txt", force_create=True)
new_file.write_text("note")
```

## Bulk Metadata Writers

GGSet includes two bulk writer families:

- CSV writers for tabular metadata
- JSON writers for dictionary-shaped metadata

These writers shard output by directory layer.

### CSV Bulk Writer

```python
from ggset import GGSet

ggset = GGSet("dataset", type_sep_level=2)
writer = ggset.crate_bulk_csv_writer("metrics", layer=2, cols=["score", "flag"])

for file in ggset.iterate("data"):
    value = int(file.read_text())
    writer.write_dict_row(file, {"score": value, "flag": value > 5})

writer.flush()

row = writer.read_for_file(file)
print(row)
```

CSV files include a `Filename` column plus the columns you specify.

### JSON Bulk Writer

```python
from ggset import GGSet

ggset = GGSet("dataset", type_sep_level=2)
writer = ggset.crate_bulk_json_writer("metadata", layer=2)

for file in ggset.iterate("data"):
    writer.write_dict_row(file, {"split": "train", "quality": "ok"})

writer.flush()

row = writer.read_for_file(file)
print(row)
```

JSON bulk files store filename keys mapped to row dictionaries.

### `save_rel_paths`

If `save_rel_paths=True`, bulk files store paths relative to the bulk file location.
When reading back combined data, GGSet normalizes those paths back to root-relative
paths.

```python
writer = ggset.crate_bulk_csv_writer(
    "metrics",
    layer=2,
    cols=["score"],
    save_rel_paths=True,
)
```

## File Helpers

`GGFile` supports these common readers:

- `read_text()`
- `read_image()`
- `read_json()`
- `read_yaml()`
- `read_dataframe()`
- `read_single_bool()`
- `read_single_int()`
- `read_single_float()`
- `read_int_list()`
- `read_float_list()`
- `read_bool_list()`

It also supports direct writing for text and images:

- `write_text(content)`
- `write_image(image)`

## Filters

You can restrict traversal at a specific level.

```python
ggset.add_filter_allow_only(2, "db1")

for file in ggset.iterate("data"):
    print(file.rel_path)
```

Or exclude directories:

```python
ggset.add_filter_exclude(2, "db2")
```
