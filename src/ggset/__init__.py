"""GGDir package."""

__all__ = [
    "__version__",
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
__version__ = "0.5.1"

from .ggset import *
