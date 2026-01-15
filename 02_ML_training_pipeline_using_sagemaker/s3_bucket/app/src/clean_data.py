# src/clean_data.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer

input_file = "/opt/ml/processing/input/read_data.csv"
output_file = "/opt/ml/processing/output/clean_data.csv"

df = pd.read_csv(input_file)

# Impute missing values: median cho numerical
imputer = SimpleImputer(strategy="median")
df[df.columns] = imputer.fit_transform(df)

# Loại bỏ giá nhà âm hoặc 0 (nếu có)
df = df[df["MedHouseVal"] > 0]

os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
