# src/prepare_data.py
import os
import pandas as pd

input_file = "/opt/ml/processing/input/clean_data.csv"
output_file = "/opt/ml/processing/output/prepared_data.csv"

df = pd.read_csv(input_file)

# Tách target
target = "MedHouseVal"
features = [c for c in df.columns if c != target]

X = df[features]
y = df[target]

# Gộp lại để lưu (SageMaker training step sẽ load toàn bộ)
df_prepared = X.copy()
df_prepared[target] = y

os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_prepared.to_csv(output_file, index=False)
print(f"Prepared data saved to {output_file}")
