# src/normalize_data.py
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

input_file = "/opt/ml/processing/input/prepared_data.csv"
output_file = "/opt/ml/processing/output/normalized_data.csv"

df = pd.read_csv(input_file)

target = "MedHouseVal"
features = [c for c in df.columns if c != target]

scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_scaled.to_csv(output_file, index=False)
print(f"Normalized data saved to {output_file}")
