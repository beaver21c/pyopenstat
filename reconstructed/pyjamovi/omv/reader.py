"""
pyjamovi OMV Reader
Deconstructs .omv ZIP archives, parses data.bin column streams, and builds DataFrame.
"""

import os
import io
import json
import struct
import zipfile
from zipfile import ZipFile
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from .types import ColumnType, DataType, MeasureType, INT_MISSING
from .dataset import JamoviDataset, JamoviColumn


def _read_null_terminated_string(buffer: bytes, offset: int) -> str:
    """Read a null-terminated UTF-8 string from a byte buffer starting at offset."""
    if offset < 0 or offset >= len(buffer):
        return ""
    end = buffer.find(b"\x00", offset)
    if end == -1:
        end = len(buffer)
    try:
        return buffer[offset:end].decode("utf-8", errors="replace")
    except Exception:
        return ""


def read_omv(path: str) -> Tuple[pd.DataFrame, JamoviDataset]:
    """
    Read a jamovi .omv file.
    Returns:
        (df, dataset): A tuple of (pandas DataFrame, JamoviDataset with complete metadata)
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OMV file not found: {path}")

    with ZipFile(path, "r") as zf:
        namelist = zf.namelist()

        # 1. Read manifest / meta
        manifest_text = ""
        if "meta" in namelist:
            manifest_text = zf.read("meta").decode("utf-8", errors="replace")
        elif "META-INF/MANIFEST.MF" in namelist:
            manifest_text = zf.read("META-INF/MANIFEST.MF").decode("utf-8", errors="replace")

        # 2. Read metadata.json
        if "metadata.json" not in namelist:
            raise ValueError(f"Corrupt OMV file: metadata.json missing in {path}")
        
        meta_dict = json.loads(zf.read("metadata.json").decode("utf-8"))
        dataset_meta = meta_dict.get("dataSet", {})
        row_count = dataset_meta.get("rowCount", 0)
        fields_meta = dataset_meta.get("fields", [])

        # 3. Read xdata.json (categorical labels)
        xdata_dict = {}
        if "xdata.json" in namelist:
            try:
                xdata_dict = json.loads(zf.read("xdata.json").decode("utf-8"))
            except Exception:
                xdata_dict = {}

        # 4. Read strings.bin if present
        strings_buf = b""
        if "strings.bin" in namelist:
            strings_buf = zf.read("strings.bin")

        # 5. Read data.bin
        if "data.bin" not in namelist:
            raise ValueError(f"Corrupt OMV file: data.bin missing in {path}")
        
        data_buf = zf.read("data.bin")

        # Construct Dataset
        dataset = JamoviDataset()
        dataset.row_count = row_count
        dataset.results_language = meta_dict.get("document", {}).get("resultsLanguage", "en")
        dataset.weights = dataset_meta.get("weights")

        df_data = {}
        data_stream = io.BytesIO(data_buf)

        for meta_col in fields_meta:
            name = meta_col.get("name", "")
            col_id = meta_col.get("id", 0)
            col_type = ColumnType.parse(meta_col.get("columnType", "Data"))
            data_type = DataType.parse(meta_col.get("dataType", "Decimal"))
            measure_type = MeasureType.parse(meta_col.get("measureType", "Continuous"))

            # Labels from xdata
            col_labels = []
            if name in xdata_dict and "labels" in xdata_dict[name]:
                col_labels = xdata_dict[name]["labels"]

            column = JamoviColumn(
                name=name,
                id=col_id,
                column_type=col_type,
                data_type=data_type,
                measure_type=measure_type,
                formula=meta_col.get("formula", ""),
                description=meta_col.get("description", ""),
                missing_values=meta_col.get("missingValues", []),
                width=meta_col.get("width", 100),
                labels=col_labels
            )
            dataset.fields.append(column)

            # Read binary column data
            if data_type == DataType.DECIMAL:
                # 8 bytes per row, <d
                bytes_to_read = row_count * 8
                col_bytes = data_stream.read(bytes_to_read)
                if len(col_bytes) == bytes_to_read:
                    vals = struct.unpack(f"<{row_count}d", col_bytes)
                    df_data[name] = [np.nan if np.isnan(v) else v for v in vals]
                else:
                    df_data[name] = [np.nan] * row_count
            elif data_type == DataType.TEXT and measure_type == MeasureType.ID:
                # 4 bytes per row, <i -> offset in strings.bin
                bytes_to_read = row_count * 4
                col_bytes = data_stream.read(bytes_to_read)
                if len(col_bytes) == bytes_to_read:
                    offsets = struct.unpack(f"<{row_count}i", col_bytes)
                    vals = []
                    for off in offsets:
                        if off == INT_MISSING:
                            vals.append(None)
                        else:
                            vals.append(_read_null_terminated_string(strings_buf, off))
                    df_data[name] = vals
                else:
                    df_data[name] = [None] * row_count
            else:
                # Integer / Categorical
                bytes_to_read = row_count * 4
                col_bytes = data_stream.read(bytes_to_read)
                if len(col_bytes) == bytes_to_read:
                    raw_ints = struct.unpack(f"<{row_count}i", col_bytes)
                    label_map = column.get_label_map()
                    vals = []
                    for code in raw_ints:
                        if code == INT_MISSING:
                            vals.append(np.nan if measure_type == MeasureType.CONTINUOUS else None)
                        elif label_map and code in label_map:
                            vals.append(label_map[code])
                        else:
                            vals.append(code)
                    df_data[name] = vals
                else:
                    df_data[name] = [None] * row_count

        df = pd.DataFrame(df_data)
        dataset._df = df.copy()
        return df, dataset
