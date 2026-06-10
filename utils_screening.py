import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


# TABLEAU_EXTENDED = [
#     "#4E79A7",  # blue
#     "#F28E2B",  # orange
#     "#E15759",  # red
#     "#76B7B2",  # teal
#     "#59A14F",  # green
#     "#EDC948",  # yellow
#     "#B07AA1",  # purple
#     "#FF9DA7",  # pink
#     "#9C755F",  # brown
#     "#BAB0AC",  # gray
#     "#999933",  # ///
# ]


"""
Preprocessing
"""
def get_included_df(path=r"data\src\conflated_database___2026_03_31__11_14_35.xlsx"):

    title_df = pd.read_excel(path, sheet_name="TitleReading")
    abstract_df = pd.read_excel(path, sheet_name="AbstractReading")
    excluded_df = pd.read_excel(path, sheet_name="TextReading_Excluded")
    included_df = pd.read_excel(path, sheet_name="TextReading_Included")

    excluded_df = excluded_df[excluded_df["Exclusion reason"].notna()].reset_index(drop=True)
    included_df = included_df[included_df["topic"].notna()].reset_index(drop=True)
    included_df.Year = included_df.Year.astype(int)

    screened = title_df.is_included.shape[0]
    print(f"screened: {screened}")

    included_titles = title_df.is_included.value_counts()["yes"].item()
    excluded_titles = screened - included_titles
    print(f"excluded -- title reading: {excluded_titles}")

    excluded_abstracts = abstract_df[abstract_df.topic.isin(["Irrelevant", "Duplicate"])].shape[0]
    included_abstracts = abstract_df.shape[0] - excluded_abstracts
    print(f"excluded -- abstract reading: {excluded_abstracts}")
    print(f"sought for retrieval (included): {included_abstracts}")
    print(f"check: {excluded_titles + excluded_abstracts + included_abstracts} =? {screened}")
    print()

    not_retrieved = excluded_df[excluded_df["Exclusion reason"] == "Not retrievable"].shape[0]
    retrieved_excluded = excluded_df[excluded_df["Exclusion reason"] != "Not retrievable"].shape[0]
    retrieved_included = included_df.shape[0]
    retrieved = retrieved_included + retrieved_excluded 

    print(f"not retrieved: {not_retrieved}")
    print(f"retrieved: {retrieved}")
    print()

    print(f"retrieved excluded: {retrieved_excluded}")
    print(f"retrieved included: {retrieved_included}")
    print(f"check: {included_abstracts} =? {not_retrieved + retrieved_excluded + retrieved_included}")
    print()

    print(excluded_df["Exclusion reason"].value_counts())

    return included_df



"""
Plots
"""
def plot_venn_diagram(included_df):

    venn_numbers = pd.DataFrame(included_df.topic.value_counts()).T

    fig, ax = plt.subplots(figsize=(8,8))

    v = venn3(
        subsets=(
            venn_numbers["Model"].item(),
            venn_numbers["Dataset"].item(),
            venn_numbers["Model, Dataset"].item(),
            venn_numbers["Evaluation"].item(),
            venn_numbers["Model, Evaluation"].item(),
            venn_numbers["Dataset, Evaluation"].item(),
            venn_numbers["Model, Dataset, Evaluation"].item()
        ),
        set_labels=('Model', 'Dataset', 'Evaluation')
    )

    # -----------------------------
    # Set label font sizes
    # -----------------------------
    for text in v.set_labels:
        text.set_fontsize(16)

    # -----------------------------
    # Subset number font sizes
    # -----------------------------
    for text in v.subset_labels:
        if text:
            text.set_fontsize(14)

    # -----------------------------
    # Region colors
    # IDs:
    # 100, 010, 110, 001, 101, 011, 111
    # -----------------------------
    v.get_patch_by_id('100').set_color("red")
    v.get_patch_by_id('010').set_color("green")
    v.get_patch_by_id('001').set_color("blue")

    # Optional:
    v.get_patch_by_id('111').set_color("purple")

    # -----------------------------
    # Transparency
    # -----------------------------
    for region in ['100', '010', '110', '001', '101', '011', '111']:
        patch = v.get_patch_by_id(region)
        if patch:
            patch.set_alpha(0.6)

    # -----------------------------
    # Borders
    # -----------------------------
    for region in ['100', '010', '110', '001', '101', '011', '111']:
        patch = v.get_patch_by_id(region)
        if patch:
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)

    # plt.show()

    plt.savefig(
        "data/results/venn.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_publisher_per_year(included_df):
    counts = included_df.groupby(['Year', 'Publisher']).size().unstack(fill_value=0).sort_index()

    totals = counts.sum(axis=0)

    sorted_publishers = (
        totals
        .sort_values(ascending=False)
        .index
    )
    counts = counts[sorted_publishers]

    counts.columns = [
        f"{publisher} ({totals[publisher]})"
        for publisher in counts.columns
    ]

    cmap = plt.cm.tab20
    num_colors = len(counts.columns)
    colors = [
        cmap(i / num_colors) for i in range(num_colors)
    ]
    # colors = TABLEAU_EXTENDED

    ax = counts.plot(
        kind='bar',
        stacked=True,
        edgecolor="white",
        linewidth=0.4,
        rot=0,
        ylabel="# papers",
        color=colors,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    
    ax.legend(
        title="Publishing Organization",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False,
    )


# def plot_publisher(included_df):
#     counts = included_df["Publisher"].value_counts(ascending=True)

#     cmap = plt.cm.tab20
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ] 

#     ax = counts.plot(kind="barh", color=colors)
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.set_xlabel("# papers")
#     ax.set_ylabel("Publisher")


# def plot_year(included_df):
#     counts = included_df["Year"].value_counts().sort_index()

#     # cmap = plt.cm.tab20
#     # num_colors = len(counts.index)
#     # colors = [
#     #     cmap(i / num_colors) for i in range(num_colors)
#     # ] 

#     ax = counts.plot(
#         kind="bar", 
#         rot=0,
#         # color=colors
#     )
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.set_xlabel("Year")
#     ax.set_ylabel("# papers")
