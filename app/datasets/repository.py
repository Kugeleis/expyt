"""Dataset repository protocol and implementations.

The ``DatasetRepository`` protocol defines the interface for loading datasets.
Concrete implementations handle the actual I/O (CSV files, databases, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import pyarrow.parquet as pq

from app.datasets.models import ColumnInfo, DatasetInfo


class DatasetRepository(Protocol):
    """Protocol for dataset loading and introspection."""

    def list_datasets(self) -> list[DatasetInfo]:
        """List all available datasets."""
        ...  # pragma: no cover

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a dataset by its ID.

        Raises:
            KeyError: If the dataset is not found.
        """
        ...  # pragma: no cover

    def get_schema(self, dataset_id: str) -> DatasetInfo:
        """Return schema metadata for a dataset.

        Raises:
            KeyError: If the dataset is not found.
        """
        ...  # pragma: no cover


@runtime_checkable
class FileLoader(Protocol):
    """Protocol for loaders of specific file formats."""

    @property
    def extension(self) -> str:
        """Return the suffix this loader handles (e.g. '.csv')."""
        ...  # pragma: no cover

    def load(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from the specified file path."""
        ...  # pragma: no cover

    def get_schema(self, path: Path) -> list[ColumnInfo]:
        """Return the column schema for the file path without reading all rows."""
        ...  # pragma: no cover


class CsvLoader:
    """Loader for CSV files."""

    @property
    def extension(self) -> str:
        """Return the file extension handled by this loader."""
        return ".csv"

    def load(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from a CSV file."""
        return pd.read_csv(path)

    def get_schema(self, path: Path) -> list[ColumnInfo]:
        """Read the schema of a CSV file using nrows=100 to infer dtypes."""
        df = pd.read_csv(path, nrows=100)
        return [
            ColumnInfo(
                name=str(col),
                dtype=str(df[col].dtype),
                nullable=True,
            )
            for col in df.columns
        ]


class XptLoader:
    """Loader for SAS XPT files."""

    @property
    def extension(self) -> str:
        """Return the file extension handled by this loader."""
        return ".xpt"

    def load(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from a SAS XPT file, decoding bytes."""
        df = pd.read_sas(path, format="xport")
        # Decode byte columns
        df.columns = pd.Index(
            [
                col.decode("utf-8") if isinstance(col, bytes) else str(col)
                for col in df.columns
            ]
        )
        # Decode string data values if they are bytes
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )
        return df

    def get_schema(self, path: Path) -> list[ColumnInfo]:
        """Read the schema of a SAS XPT file, decoding bytes."""
        df = self.load(path)
        return [
            ColumnInfo(
                name=str(col),
                dtype=str(df[col].dtype),
                nullable=True,
            )
            for col in df.columns
        ]


class HdfLoader:
    """Loader for HDF5 files."""

    @property
    def extension(self) -> str:
        """Return the file extension handled by this loader."""
        return ".h5"

    def load(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from an HDF5 file."""
        with pd.HDFStore(str(path), mode="r") as store:
            keys = store.keys()
            if not keys:
                msg = f"HDF5 file {path.name} contains no keys."
                raise ValueError(msg)
            df = store.get(keys[0])
            if not isinstance(df, pd.DataFrame):
                msg = f"Key {keys[0]} in HDF5 file {path.name} is not a DataFrame."
                raise TypeError(msg)
            return df

    def get_schema(self, path: Path) -> list[ColumnInfo]:
        """Read the schema of an HDF5 file."""
        with pd.HDFStore(str(path), mode="r") as store:
            keys = store.keys()
            if not keys:
                msg = f"HDF5 file {path.name} contains no keys."
                raise ValueError(msg)
            first_key = keys[0]
            try:
                # Try selecting 0 rows if stored in table format
                df = store.select(first_key, start=0, stop=1)
            except TypeError:
                # Fallback for fixed format files
                df = store.get(first_key)
            if not isinstance(df, pd.DataFrame):
                msg = f"Key {first_key} in HDF5 file {path.name} is not a DataFrame."
                raise TypeError(msg)
            df = df.head(1)
        return [
            ColumnInfo(
                name=str(col),
                dtype=str(df[col].dtype),
                nullable=True,
            )
            for col in df.columns
        ]


class ParquetLoader:
    """Loader for Parquet files."""

    @property
    def extension(self) -> str:
        """Return the file extension handled by this loader."""
        return ".parquet"

    def load(self, path: Path) -> pd.DataFrame:
        """Load a DataFrame from a Parquet file."""
        return pd.read_parquet(path)

    def get_schema(self, path: Path) -> list[ColumnInfo]:
        """Read the schema of a Parquet file using pyarrow.read_schema."""
        schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        df = schema.empty_table().to_pandas()
        return [
            ColumnInfo(
                name=str(col),
                dtype=str(df[col].dtype),
                nullable=True,
            )
            for col in df.columns
        ]


class MultiFormatDatasetRepository:
    """Loads datasets from various file formats in a configured directory.

    Uses registered FileLoader instances to load files of different extensions.
    """

    def __init__(self, data_dir: Path, loaders: list[FileLoader] | None = None) -> None:
        """Initialize the repository.

        Args:
            data_dir: Path to the directory containing dataset files.
            loaders: Optional list of file loaders. Defaults to built-in loaders.
        """
        self._data_dir = data_dir
        if loaders is None:
            loaders = [CsvLoader(), XptLoader(), HdfLoader(), ParquetLoader()]
        self._loaders = {loader.extension.lower(): loader for loader in loaders}

    def _get_loader_and_path(self, dataset_id: str) -> tuple[FileLoader, Path]:
        """Find the file and its loader for a dataset ID.

        Raises:
            KeyError: If the dataset is not found.
        """
        if not self._data_dir.exists():
            msg = f"Data directory {self._data_dir} does not exist"
            raise KeyError(msg)
        for path in self._data_dir.iterdir():
            if path.is_file() and path.stem == dataset_id:
                ext = path.suffix.lower()
                if ext in self._loaders:
                    return self._loaders[ext], path
        msg = f"Dataset {dataset_id!r} not found or format not supported."
        raise KeyError(msg)

    def list_datasets(self) -> list[DatasetInfo]:
        """List all datasets in the configured directory matching loader extensions."""
        datasets: list[DatasetInfo] = []
        if not self._data_dir.exists():
            return datasets
        added_stems: set[str] = set()
        for path in sorted(self._data_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in self._loaders:
                stem = path.stem
                if stem not in added_stems:
                    added_stems.add(stem)
                    ext_upper = path.suffix.upper()[1:]
                    datasets.append(
                        DatasetInfo(
                            id=stem,
                            name=stem.replace("_", " ").title(),
                            description=f"{ext_upper} dataset from {path.name}",
                        )
                    )
        return datasets

    def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load a dataset by its ID."""
        loader, path = self._get_loader_and_path(dataset_id)
        return loader.load(path)

    def get_schema(self, dataset_id: str) -> DatasetInfo:
        """Return schema metadata for a dataset."""
        loader, path = self._get_loader_and_path(dataset_id)
        columns = loader.get_schema(path)
        ext_upper = path.suffix.upper()[1:]
        return DatasetInfo(
            id=dataset_id,
            name=dataset_id.replace("_", " ").title(),
            description=f"{ext_upper} dataset from {path.name}",
            columns=columns,
        )


class CsvDatasetRepository(MultiFormatDatasetRepository):
    """Backward-compatible repository that defaults to CSV files only."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize the CSV repository."""
        super().__init__(data_dir, loaders=[CsvLoader()])
