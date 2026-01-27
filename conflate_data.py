# py conflate_data.py

from _utils import papers_plus_preprints

preprints_paths = [
    "data\src\preprints_1_200.csv",
    "data\src\preprints_201_230.csv",
]

papers_path = "data\src\scopus_export_Jan 27-2026_63680511-6fe7-4ee6-bfb0-5a80574d6795.csv"

papers_plus_preprints(papers_path, preprints_paths)