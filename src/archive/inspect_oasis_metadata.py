from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
file = root / "data_audit/oasis_cross-sectional.xlsx"

df = pd.read_excel(file)

print("Rows:", len(df))
print("Columns:", list(df.columns))
print()
print(df.head(10).to_string(index=False))