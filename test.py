# py test.py

import pandas as pd

df = pd.read_csv("data\src\preprints_1_200.csv", sep=",", encoding="")
# df = pd.read_csv("data\src\scopus_export_Jan 27-2026_63680511-6fe7-4ee6-bfb0-5a80574d6795.csv")
print(df.head(5))