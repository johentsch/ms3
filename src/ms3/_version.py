"""The version of the installed ms3 distribution."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ms3")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
