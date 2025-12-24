# src/read_data.py
import os
import pandas as pd

input_path = "/opt/ml/processing/input"
output_path = "/opt/ml/processing/output"

# Giả sử file CSV trong input folder
csv_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
if len(csv_files) == 0:
    raise FileNotFoundError("Không tìm thấy file CSV trong input folder")

df_list = [pd.read_csv(os.path.join(input_path, f)) for f in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Lưu ra output
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, "house_price.csv")
df.to_csv(output_file, index=False)
print(f"Read data saved to {output_file}")
