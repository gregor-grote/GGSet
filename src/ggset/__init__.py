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
    "GGBulkCsvFileCollection",
    "GGBulkJsonFileCollection",
]
__version__ = "0.3.0"

from .ggset import *
