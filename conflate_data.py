# py conflate_data.py
# python conflate_data.py

from _utils import papers_plus_preprints

preprints_paths = [
    "data\src\preprints_v06.csv",
]

papers_path = "data\src\scopus_export_Jan 29-2026_791ef687-8ddb-44b6-95e0-193bb5143d8f.csv"

papers_plus_preprints(papers_path, preprints_paths)