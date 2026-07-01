# GGSet

A lightweight library for working with structured datasets where data and labels are stored across flexible directory layouts.

## Core Idea

`GGSet` provides a unified interface to iterate over files and access corresponding metadata (labels) — regardless of how the dataset is organized on disk.

It is especially useful for:

* Image datasets with separate or distributed annotations
* Machine learning pipelines with custom folder structures
* Datasets with CSV or JSON metadata files

## Installation

```bash
pip install git+https://github.com/gregor-grote/GGSet.git
```

---

## Main Features

### 1. Branched Dataset Support (Core Feature)

Work with datasets where data and labels live in different directory branches.

* Configure via `data_type_level`
* Access corresponding files across branches

```python
ggset = GGSet(path, data_type_level=2)

for file in ggset.iterate(data_type="images"):
    label_file = file.get_corresponding_file(data_type="labels", extension=".json")
```

Supports:

* Reading corresponding label files
* Writing new labels into parallel branches
* Automatic path resolution between branches

---

### 2. CSV Metadata Collections

Read and write labels stored in CSV files across the dataset.

```python
with ggset.create_bulk_csv_collection("labels.csv", layer=2, rel_paths=True) as labels:
    for file, label in labels:
        ...
```

Features:

* Iterate `(file, label)` pairs
* Load full dataset as a pandas DataFrame
* Write new metadata entries
* Optional caching for performance vs. memory trade-off

---

### 3. JSON Metadata Collections

Same concept as CSV, but for JSON-based metadata.

```python
with ggset.create_bulk_json_collection("labels.json", layer=2, rel_paths=True) as labels:
    for file, label in labels:
        ...
```

Features:

* Read full metadata as dictionary
* Seamless integration with dataset iteration
* Flexible schema handling

---

### 4. Corresponding File Handling

Easily resolve related files based on dataset structure:

* Across branches:

```python
file.get_corresponding_file(data_type="labels", extension=".json")
```

* In the same directory:

```python
file.get_corresponding_file_in_same_dir(".json")
```

Supports:

* Reading / writing labels
* Automatic creation of missing files (`force_create=True`)

---

### 5. Corresponding Directory Handling

Work with label directories instead of single files.

* Same directory:

```python
file.get_corresponding_dir_in_same_dir()
```

* Different branch:

```python
file.get_corresponding_dir("labels")
```

Features:

* Iterate over label files per sample
* Create new label directories dynamically
* Auto-generate files inside label folders

---

## Supported Dataset Layouts

GGSet works with multiple dataset structures:

* **Branched datasets** (data and labels separated)
* **Flat datasets with CSV/JSON metadata**
* **Same-directory labels** (image + annotation file)
* **Per-sample label directories**
* **Label directories in separate branches**

---

## Minimal Example

```python
from ggset import GGSet

ggset = GGSet("dataset_path")

for file in ggset.iterate(filter_endings=(".png",)):
    img = file.read_image()
```

---

## Summary

GGSet abstracts away dataset structure complexity and provides:

* Unified iteration over files
* Flexible metadata handling (CSV / JSON)
* Robust mapping between data and labels
* Support for complex real-world dataset layouts
