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
__version__ = "0.4.0"

from .ggset import *
