# py test.py

import pandas as pd
from _utils import dct2excel


path_main = r"data\result\conflated_database___2026_01_27__19_12_57.xlsx"""
path_add = r"data\labeled_before\final_database - 03_TitleReview.csv"""

df_dct = pd.read_excel(path_main, sheet_name=None)
df_main = df_dct["TitleReading"]
df_add = pd.read_csv(path_add)[["Title", "Decision"]]
df_add.loc[df_add.Decision != "yes", "Decision"] = "no"

df = pd.merge(df_main, df_add, on="Title", how="left")
print("yes:", df[df.Decision == "yes"].shape[0])
print("no:", df[df.Decision == "no"].shape[0])
print("empty:", df[~df.Decision.isin(["yes", "no"])].shape[0])
df_dct["TitleReading_"] = df

dct2excel(df_dct, path_main)