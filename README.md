# GGSet

GGSet is a small utility library for working with datasets stored in directory structures. It provides a consistent way to navigate files, map related data across folders, and manage metadata stored in JSON or CSV files.

The goal is to remove boilerplate when dealing with datasets that are not strictly flat or standardized.

---

## Installation

```bash
pip install -U git+https://github.com/gregor-grote/GGSet.git
```

---

## Basic Usage

```python
from ggset import GGSet

ggset = GGSet("path/to/dataset")

for file in ggset.iterate(filter_endings=(".png",)):
    image = file.read_image()
```

`GGSet` represents the root of your dataset. From there, you can iterate through files, access directories, and resolve related files.

---

## Working with Structured Datasets

### Separate Data Branches

If your dataset separates data into branches (for example `images/` and `labels/`), you can configure this using `data_type_level`.

```text
dataset/
    train/
        images/
            sample1.png
        labels/
            sample1.json
```

```python
ggset = GGSet("dataset", data_type_level=2)

for img_file in ggset.iterate(data_type="images"):
    label_file = img_file.get_corresponding_file(
        data_type="labels",
        extension=".json"
    )
    label_data = label_file.read_json()
```

The relative structure is preserved when resolving corresponding files.

---

### Same-Directory Files

If files live next to each other:

```text
sample1.png
sample1.json
```

```python
label_file = file.get_corresponding_file_in_same_dir(".json")
```

---

### Per-Sample Directories

```text
sample1.png
sample1/
    label1.json
    label2.json
```

```python
label_dir = file.get_corresponding_dir_in_same_dir()

for label_file in label_dir.get_sub_files():
    data = label_file.read_json()
```

---

## Bulk Metadata (CSV / JSON)

GGSet supports reading and writing metadata stored in shared files (for example one CSV per folder).

### CSV Collections
```text
train/
	annotations.csv
	images/
		sample1.png
		sample2.png
test/
	annotations.csv
	images/
		sample3.png
```


```python
with ggset.create_bulk_csv_collection(
    "annotations.csv",
    layer=2,
    rel_paths=True,
    caching=True
) as collection:

    for file, row in collection:
        ...

    collection.write(file, {"label": 1})
```

Key behavior:

* Iteration yields `(file, data_dict)`
* Files are resolved automatically
* Writes are buffered and flushed on exit

Options:

* `layer`: the directory level where the CSV files are located (1 = below root, below each subdirectory below root, etc.)
* `rel_paths`: store paths relative to the CSV location (`True`) or to the dataset root (`False`)
* `caching=True`: keeps data in memory and overwrites existing rows
* `caching=False`: appends rows (duplicates possible, but faster for large datasets)

---

### JSON Collections

```text
train/
	annotations.json
	images/
		sample1.png
		sample2.png
test/
	annotations.json
	images/
		sample3.png
```

```python
collection = ggset.create_bulk_json_collection(
    "annotations.json",
    layer=2,
    rel_paths=False
)

collection.write(file, {"label": 1})
data = collection.read_dict()
```

JSON collections store data as:

```json
{
  "path/to/file": { ... }
}
```

Options:

* `layer`: the directory level where the JSON files are located (1 = below root, below each subdirectory below root, etc.)
* `rel_paths`: store paths relative to the JSON location (`True`) or to the dataset root (`False`)

---

## File API

Each file is represented by a `GGFile` object.

### Reading

```python
file.read_image()
file.read_json()
file.read_yaml()
file.read_text()
file.read_dataframe()
file.read_np_array()
```

There are also helpers for simple values:

```python
file.read_single_int()
file.read_single_float()
file.read_single_bool()
```

---

### Writing

```python
file.write_json(data)
file.write_text("hello")
file.write_dataframe(df)
file.write_image(img)
file.write_np_array(arr)
```

Files are created automatically if they do not exist.

---

## Directory Navigation

You can navigate the dataset structure explicitly:

```python
root = GGSet("dataset")

train_dir = root.get_sub_dir("train")
images_dir = train_dir.get_sub_dir("images")

file = images_dir.get_file("sample1.png")
```

Useful helpers:

* `exists()`
* `touch()` (create directories or files)
* `file_count()`
* `iterate_layer(level)`

---

## Iteration Behavior

`iterate()` adapts to the dataset structure:

```python
ggset.iterate(data_type="images")
ggset.iterate(filter_endings=(".png",))
```

It will:

* Traverse directories recursively
* Respect `data_type_level`
* Filter by file endings if provided

---

## Filtering Directories

You can restrict traversal:

```python
ggset.add_filter_allow_only(level=1, "train", "test")
ggset.add_filter_exclude(level=2, "tmp")
```

Filters apply per directory level.

Note, that when creating a custom `GetSubDirsStrategy`, the filters are not applied anymore. See [example-usage.ipynb](example-usage.ipynb) for more details.

---

## Notes

* Paths are resolved lazily; the directory tree is not built upfront
* Non-existing files and their parent directories are only created when writing or explicitly when calling `touch()`
* Bulk collections can target a different output dataset via `bulk_files_root`, so you can save the annotations to a different location than the data files.
* This library is designed to be flexible and lightweight, without enforcing a specific dataset structure by relying extensively on the Strategy pattern. (See [example-usage.ipynb](example-usage.ipynb) for more details.)

---

## Summary

GGSet is useful when:

* Your dataset is not flat
* Data and labels are stored separately
* You need consistent file mapping
* You want lightweight metadata handling without a full database

It keeps everything file-based while still providing structure.
