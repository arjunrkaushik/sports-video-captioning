import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os

parser = argparse.ArgumentParser()
parser.add_argument("csv_dir", type=str)
parser.add_argument("train_split", type=float, default=0.8)
args = parser.parse_args()

# Load the CSV file
csv_file_path = os.path.join(args.csv_dir, 'Labels-caption.csv')
df = pd.read_csv(csv_file_path)

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=1 - args.train_split, random_state=42)

# Save the split data to new CSV files
train_df.to_csv(os.path.join(args.csv_dir,'train.csv'), index=False)
val_df.to_csv(os.path.join(args.csv_dir,'val.csv'), index=False)
