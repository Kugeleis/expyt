"""Unit tests for the dataset repositories and format loaders."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from app.datasets.models import DatasetInfo
from app.datasets.repository import (
    CsvDatasetRepository,
    CsvLoader,
    HdfLoader,
    MultiFormatDatasetRepository,
    ParquetLoader,
    XptLoader,
)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a sample CSV."""
    csv_content = "group,value\nA,1.0\nA,2.0\nB,3.0\nB,4.0\n"
    (tmp_path / "sample.csv").write_text(csv_content)
    return tmp_path


@pytest.fixture
def repo(data_dir: Path) -> CsvDatasetRepository:
    """Provide a CsvDatasetRepository pointing at the temp dir."""
    return CsvDatasetRepository(data_dir)


def test_list_datasets(repo: CsvDatasetRepository) -> None:
    """list_datasets returns metadata for each CSV file."""
    datasets = repo.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].id == "sample"
    assert isinstance(datasets[0], DatasetInfo)


def test_list_datasets_empty_dir(tmp_path: Path) -> None:
    """list_datasets returns empty list for directory with no CSVs."""
    repo = CsvDatasetRepository(tmp_path)
    assert repo.list_datasets() == []


def test_list_datasets_missing_dir(tmp_path: Path) -> None:
    """list_datasets returns empty list for non-existent directory."""
    repo = CsvDatasetRepository(tmp_path / "nonexistent")
    assert repo.list_datasets() == []


def test_load_dataset(repo: CsvDatasetRepository) -> None:
    """load_dataset returns a DataFrame with expected shape."""
    df = repo.load_dataset("sample")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["group", "value"]
    assert len(df) == 4


def test_load_dataset_missing_raises(repo: CsvDatasetRepository) -> None:
    """load_dataset raises KeyError for missing dataset."""
    with pytest.raises(KeyError, match="not found"):
        repo.load_dataset("nonexistent")


def test_get_schema(repo: CsvDatasetRepository) -> None:
    """get_schema returns column metadata."""
    schema = repo.get_schema("sample")
    assert schema.id == "sample"
    assert len(schema.columns) == 2
    assert schema.columns[0].name == "group"
    assert schema.columns[1].name == "value"


def test_get_schema_missing_raises(repo: CsvDatasetRepository) -> None:
    """get_schema raises KeyError for missing dataset."""
    with pytest.raises(KeyError, match="not found"):
        repo.get_schema("nonexistent")


def test_csv_loader(tmp_path: Path) -> None:
    """Test CsvLoader loads and parses schemas correctly."""
    loader = CsvLoader()
    assert loader.extension == ".csv"

    csv_path = tmp_path / "test.csv"
    csv_path.write_text("x,y\n1,a\n2,b\n")

    df = loader.load(csv_path)
    assert list(df.columns) == ["x", "y"]
    assert len(df) == 2

    schema = loader.get_schema(csv_path)
    assert len(schema) == 2
    assert schema[0].name == "x"
    assert schema[1].name == "y"


def test_xpt_loader() -> None:
    """Test XptLoader loads SAS XPT file and decodes byte data."""
    loader = XptLoader()
    assert loader.extension == ".xpt"

    # read_sas returns byte columns and byte rows
    mock_df = pd.DataFrame(
        {
            b"GROUP": [b"A", b"B"],
            b"VALUE": [10.0, 20.0],
        }
    )

    with patch("pandas.read_sas") as mock_read:
        mock_read.return_value = mock_df

        df = loader.load(Path("dummy.xpt"))
        assert list(df.columns) == ["GROUP", "VALUE"]
        assert df["GROUP"].tolist() == ["A", "B"]

        schema = loader.get_schema(Path("dummy.xpt"))
        assert len(schema) == 2
        assert schema[0].name == "GROUP"
        assert schema[1].name == "VALUE"


def test_hdf_loader(tmp_path: Path) -> None:
    """Test HdfLoader handles fixed and table HDF5 layouts."""
    loader = HdfLoader()
    assert loader.extension == ".h5"

    df = pd.DataFrame({"col1": [1, 2], "col2": ["x", "y"]})

    # Fixed format
    h5_fixed = tmp_path / "fixed.h5"
    df.to_hdf(h5_fixed, key="data", format="fixed")

    assert list(loader.load(h5_fixed).columns) == ["col1", "col2"]
    with patch.object(
        pd.HDFStore,
        "select",
        side_effect=TypeError("Fixed format doesn't support select"),
    ):
        assert len(loader.get_schema(h5_fixed)) == 2

    # Table format
    h5_table = tmp_path / "table.h5"
    df.to_hdf(h5_table, key="data", format="table")

    assert list(loader.load(h5_table).columns) == ["col1", "col2"]
    assert len(loader.get_schema(h5_table)) == 2

    # Missing dataset keys check
    empty_h5 = tmp_path / "empty.h5"
    with pd.HDFStore(str(empty_h5), mode="w"):
        pass

    with pytest.raises(ValueError, match="contains no keys"):
        loader.load(empty_h5)

    with pytest.raises(ValueError, match="contains no keys"):
        loader.get_schema(empty_h5)

    # Series type check error handling
    series_h5 = tmp_path / "series.h5"
    pd.Series([1, 2]).to_hdf(series_h5, key="data")

    with pytest.raises(TypeError, match="is not a DataFrame"):
        loader.load(series_h5)

    with pytest.raises(TypeError, match="is not a DataFrame"):
        loader.get_schema(series_h5)


def test_parquet_loader(tmp_path: Path) -> None:
    """Test ParquetLoader schema extraction and data load."""
    loader = ParquetLoader()
    assert loader.extension == ".parquet"

    df = pd.DataFrame({"x": [1.5, 2.5], "y": ["a", "b"]})
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path)

    assert list(loader.load(parquet_path).columns) == ["x", "y"]

    schema = loader.get_schema(parquet_path)
    assert len(schema) == 2
    assert schema[0].name == "x"
    assert schema[1].name == "y"


def test_multi_format_repository(tmp_path: Path) -> None:
    """Test MultiFormatDatasetRepository scanning and dataset loading."""
    df = pd.DataFrame({"x": [1]})
    df.to_csv(tmp_path / "ds_csv.csv", index=False)
    df.to_parquet(tmp_path / "ds_pq.parquet")
    df.to_hdf(tmp_path / "ds_h5.h5", key="data")

    # Save unsupported extension
    df.to_csv(tmp_path / "ds_unsupported.txt", index=False)

    repo = MultiFormatDatasetRepository(tmp_path)

    # list_datasets should return only the three supported ones
    datasets = repo.list_datasets()
    assert len(datasets) == 3
    ids = {d.id for d in datasets}
    assert ids == {"ds_csv", "ds_pq", "ds_h5"}

    # load dataset
    assert repo.load_dataset("ds_pq").shape == (1, 1)

    # get schema
    schema = repo.get_schema("ds_h5")
    assert schema.id == "ds_h5"
    assert len(schema.columns) == 1
    assert schema.columns[0].name == "x"

    # Missing repository folder check
    missing_repo = MultiFormatDatasetRepository(tmp_path / "nonexistent")
    assert missing_repo.list_datasets() == []

    with pytest.raises(KeyError, match="does not exist"):
        missing_repo.load_dataset("ds_pq")
