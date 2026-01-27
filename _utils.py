import re
import datetime
import pandas as pd


"""
Utils
"""
def get_curr_time():
    return datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def dct2excel(dct, save_path):
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for sheet_name, df in dct.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


"""
Text Processing
"""
def remove_spaces(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = re.sub(r"\xa0", " ", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    return t


def text_for_deduplication(t):
    t = remove_spaces(t)
    t = t.lower()
    t = re.sub("[^a-z ]+", "", t)
    return t


def drop_duplicates(result):
    src_col = "Title"
    res_col = "Title_Deduplication"
    result[res_col] = result[src_col].apply(text_for_deduplication)
    n1 = result.shape[0]
    result = result.drop_duplicates(res_col)
    n2 = result.shape[0]
    print(f"before deduplication: {n1}")
    print(f"dropped duplicates: {n1 - n2}")
    print(f"after deduplication: {n2}")
    return result



"""
Data Conflation
"""
def preprocess_preprints(paths):
    result = pd.concat([pd.read_csv(path, sep=",") for path in paths], axis=0)
    result = result.rename(columns={"Document title": "Title"})
    result = result.rename(columns={"Repository": "Source"})
    if "Index" in result.columns:
        result = result.drop(columns=["Index"])
    return result


def papers_plus_preprints(papers_path, preprints_paths):

    # read papers
    papers = pd.read_csv(papers_path, sep=",")

    # conflate preprint chunks
    preprints = preprocess_preprints(preprints_paths)

    # conflate papers and preprints
    result = pd.concat([papers, preprints], axis=0)
    result = drop_duplicates(result)

    # save
    name = f"conflated_database___{get_curr_time()}"
    save_path = f"data/result/{name}.xlsx"
    dct = {
        name: result,
        "TitleReading": result[["Title", "Year", "Source"]],
    }
    dct2excel(dct, save_path)
    