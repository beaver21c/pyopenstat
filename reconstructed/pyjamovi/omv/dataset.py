"""
pyjamovi Dataset Structure
Represents dataset metadata, categorical labels, and observations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .types import ColumnType, DataType, MeasureType, INT_MISSING


@dataclass
class JamoviColumn:
    name: str
    id: int
    column_type: ColumnType = ColumnType.DATA
    data_type: DataType = DataType.DECIMAL
    measure_type: MeasureType = MeasureType.CONTINUOUS
    formula: str = ""
    description: str = ""
    missing_values: List[Any] = field(default_factory=list)
    width: int = 100
    # Categorical labels: list of tuples (code, label, import_value, pinned)
    labels: List[List[Any]] = field(default_factory=list)

    def get_label_map(self) -> Dict[int, str]:
        """Map internal code to human-readable label."""
        return {item[0]: str(item[1]) for item in self.labels if len(item) >= 2}

    def get_code_map(self) -> Dict[str, int]:
        """Map human-readable label to internal code."""
        return {str(item[1]): item[0] for item in self.labels if len(item) >= 2}


class JamoviDataset:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.fields: List[JamoviColumn] = []
        self.row_count: int = 0
        self.results_language: str = "en"
        self.weights: Optional[str] = None
        self._df: Optional[pd.DataFrame] = None
        
        if df is not None:
            self.from_dataframe(df)

    @property
    def column_count(self) -> int:
        return len(self.fields)

    def get_column(self, name: str) -> Optional[JamoviColumn]:
        for col in self.fields:
            if col.name == name:
                return col
        return None

    def from_dataframe(self, df: pd.DataFrame):
        """Construct JamoviDataset from pandas DataFrame with smart type inference."""
        self._df = df.copy()
        self.row_count = len(df)
        self.fields = []

        for idx, col_name in enumerate(df.columns):
            series = df[col_name]
            col_id = idx + 1
            
            # Type inference
            if pd.api.types.is_numeric_dtype(series):
                # Check if it has float/decimal values
                valid_num = series.dropna()
                is_all_int = np.all(valid_num == valid_num.astype(int)) if len(valid_num) > 0 else True
                
                if not is_all_int:
                    c_type = DataType.DECIMAL
                    m_type = MeasureType.CONTINUOUS
                    labels = []
                else:
                    unique_vals = sorted(valid_num.unique())
                    # If very few integer categories (e.g. 0/1 or 1-5), treat as nominal or ordinal
                    if len(unique_vals) <= 10:
                        c_type = DataType.INTEGER
                        m_type = MeasureType.NOMINAL
                        labels = [[int(val), str(val), int(val), True] for val in unique_vals]
                    else:
                        c_type = DataType.INTEGER
                        m_type = MeasureType.CONTINUOUS
                        labels = []
            else:
                # Text or categorical
                c_type = DataType.INTEGER
                m_type = MeasureType.NOMINAL
                unique_vals = list(pd.Series(series.dropna().unique()).astype(str))
                labels = [[i, val, val, True] for i, val in enumerate(unique_vals)]

            col = JamoviColumn(
                name=str(col_name),
                id=col_id,
                column_type=ColumnType.DATA,
                data_type=c_type,
                measure_type=m_type,
                labels=labels
            )
            self.fields.append(col)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the backing pandas DataFrame."""
        if self._df is not None:
            return self._df.copy()
        return pd.DataFrame()
