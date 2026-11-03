"""
pyjamovi OMV Writer
Packs pandas DataFrame and jamovi metadata into an authentic .omv ZIP container.
Can be opened directly in jamovi Desktop.
"""

import os
import io
import json
import struct
import zipfile
from zipfile import ZipFile
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import numpy as np

from .types import ColumnType, DataType, MeasureType, INT_MISSING
from .dataset import JamoviDataset, JamoviColumn


def write_omv(
    data: Union[pd.DataFrame, JamoviDataset],
    output_path: str,
    analyses: Optional[List[Any]] = None,
    html_report: Optional[str] = None
) -> str:
    """
    Write a pandas DataFrame or JamoviDataset to a valid jamovi .omv file.
    
    Args:
        data: DataFrame or JamoviDataset
        output_path: Filepath ending in .omv
        analyses: Optional list of BaseAnalysis results to embed
        html_report: Optional self-contained HTML report for index.html
    
    Returns:
        The written output_path
    """
    if isinstance(data, pd.DataFrame):
        dataset = JamoviDataset(data)
        df = data
    elif isinstance(data, JamoviDataset):
        dataset = data
        df = dataset.to_dataframe()
    else:
        raise TypeError("data must be a pandas DataFrame or JamoviDataset")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    row_count = len(df)
    dataset.row_count = row_count

    # Prepare string pool
    string_pool = bytearray()
    string_table_required = False

    data_bin_buffer = bytearray()
    xdata_dict: Dict[str, Any] = {}
    fields_meta: List[Dict[str, Any]] = []

    for col in dataset.fields:
        name = col.name
        col_type = col.column_type
        data_type = col.data_type
        measure_type = col.measure_type

        series = df[name] if name in df.columns else pd.Series([None] * row_count)

        # Build column metadata
        field_dict = {
            "name": name,
            "id": col.id,
            "columnType": col_type.value,
            "dataType": data_type.value,
            "measureType": measure_type.value,
            "formula": col.formula,
            "description": col.description,
            "missingValues": col.missing_values,
            "width": col.width
        }
        fields_meta.append(field_dict)

        # Build categorical mapping if needed
        if col.labels:
            xdata_dict[name] = {"labels": col.labels}
            code_map = col.get_code_map()
        else:
            code_map = {}

        # Pack binary data into data_bin_buffer
        if data_type == DataType.DECIMAL:
            for val in series:
                if pd.isna(val) or val is None:
                    d_val = float("nan")
                else:
                    try:
                        d_val = float(val)
                    except (ValueError, TypeError):
                        d_val = float("nan")
                data_bin_buffer.extend(struct.pack("<d", d_val))

        elif data_type == DataType.TEXT and measure_type == MeasureType.ID:
            string_table_required = True
            for val in series:
                if pd.isna(val) or val is None or str(val) == "":
                    offset = INT_MISSING
                else:
                    text_str = str(val)
                    offset = len(string_pool)
                    string_pool.extend(text_str.encode("utf-8") + b"\x00")
                data_bin_buffer.extend(struct.pack("<i", offset))

        else:
            # Integer or Categorical
            for val in series:
                if pd.isna(val) or val is None:
                    i_val = INT_MISSING
                else:
                    str_val = str(val)
                    if code_map and str_val in code_map:
                        i_val = int(code_map[str_val])
                    else:
                        try:
                            i_val = int(val)
                        except (ValueError, TypeError):
                            # Try float then int
                            try:
                                i_val = int(float(val))
                            except Exception:
                                i_val = INT_MISSING
                data_bin_buffer.extend(struct.pack("<i", i_val))

    # Construct metadata.json
    metadata_json = {
        "document": {
            "resultsLanguage": dataset.results_language
        },
        "dataSet": {
            "rowCount": row_count,
            "columnCount": len(dataset.fields),
            "fields": fields_meta,
            "transforms": [],
            "weights": dataset.weights
        }
    }

    # Construct meta manifest text
    manifest_lines = [
        "Manifest-Version: 1.0",
        "Data-Archive-Version: 1.0.2",
        "jamovi-Archive-Version: 11.0",
        "Created-By: pyjamovi (Pure-Python Clean-Room Engine 1.0.0)"
    ]
    manifest_bytes = ("\n".join(manifest_lines) + "\n").encode("utf-8")

    with ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. meta
        zf.writestr("meta", manifest_bytes)

        # 2. metadata.json
        zf.writestr("metadata.json", json.dumps(metadata_json, indent=2).encode("utf-8"))

        # 3. xdata.json
        zf.writestr("xdata.json", json.dumps(xdata_dict, indent=2).encode("utf-8"))

        # 4. data.bin
        zf.writestr("data.bin", bytes(data_bin_buffer))

        # 5. strings.bin if required
        if string_table_required or len(string_pool) > 0:
            zf.writestr("strings.bin", bytes(string_pool))

        # 6. index.html report
        if html_report:
            zf.writestr("index.html", html_report.encode("utf-8"))
        elif analyses:
            # Generate aggregate HTML report from analyses
            html_parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>jamovi Analysis Report</title>",
                          "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:24px;color:#2c3e50;background:#f8f9fa;}",
                          ".jamovi-card{background:#fff;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.1);padding:20px;margin-bottom:20px;}",
                          "table{border-collapse:collapse;width:100%;margin-top:12px;}th,td{border-top:1px solid #dee2e6;border-bottom:1px solid #dee2e6;padding:8px 12px;text-align:left;}th{background:#f1f3f5;font-weight:600;}",
                          "h2{margin-top:0;color:#1a365d;border-bottom:2px solid #3182ce;padding-bottom:6px;}",
                          ".footnote{font-size:0.85em;color:#6c757d;margin-top:6px;}</style></head><body>"]
            for idx, a in enumerate(analyses):
                html_parts.append(f"<div class='jamovi-card'><h2>{getattr(a, 'title', f'Analysis {idx+1}')}</h2>")
                if hasattr(a, "to_html_snippet"):
                    html_parts.append(a.to_html_snippet())
                html_parts.append("</div>")
            html_parts.append("</body></html>")
            zf.writestr("index.html", "\n".join(html_parts).encode("utf-8"))

        # 7. Serialized analyses directories
        if analyses:
            for idx, a in enumerate(analyses, start=1):
                analysis_name = getattr(a, "name", f"analysis_{idx}")
                analysis_title = getattr(a, "title", analysis_name)
                analysis_dir = f"{idx:02d} {analysis_title}/analysis"
                if hasattr(a, "serialize_state"):
                    zf.writestr(analysis_dir, a.serialize_state().encode("utf-8"))

    return output_path
