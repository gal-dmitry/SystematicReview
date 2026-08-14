import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from palettes import DEFAULT_PALETTE, MODALITY_COLORS, MODALITY_METRIC_COLORS, tab20_colors


PATH = r"data\src\conflated_database___2026_03_31__11_14_35.xlsx"


"""
Data Processing
"""
def _process_metrics(df):
    metric_cols = ["Similarity", "Instruction-Following", "Normative Compliance", "Intrinsic"]
    df_metrics = df[metric_cols]
    for col in metric_cols: 
        df_metrics[col] = df_metrics[col].apply(lambda x: x.split(", "))
    return df_metrics


def _update_model_name(name):
    if "Hybrid" in name:
        if "Discriminative" in name:
            return "Discriminative: Hybrid"
        return "Generative: Hybrid"
    else:
        if "Discriminative" in name:
            return name
        return f"Generative: {name}"


def _process_model(df_model):
    df_model["Discriminative/Generative"] = df_model["Core Generative Paradigm"].apply(lambda x: "Discriminative" if "Discriminative" in x else "Generative")
    df_model["Model"] = df_model["Core Generative Paradigm"].apply(_update_model_name)
    return df_model


def _process_input_output(df_model):
    df_model["input"] = df_model["Neural Input (modality: semantics)"].apply(lambda x: set(x.split(", ")))

    df_model["input_modality"] = df_model["input"].apply(lambda x: set(i.split(": ")[0] for i in x))
    df_model["input_content"] = df_model["input"].apply(lambda x: set(i.split(": ")[1] for i in x if i != "Absent")) ### to check later !!!

    df_model["output_modality"] = df_model["Neural Output (modality)"].apply(lambda x: x.split(", "))
    df_model["output_content"] = df_model["Ultimate Output semantics"].apply(lambda x: set(x.split(", ")))

    return df_model


def read_and_process_csv(path):

    df = pd.read_excel(path, sheet_name="TextReading_Included")
    df = df[df.topic.notna()].reset_index(drop=True)
    df.topic = df.topic.apply(lambda x: x.split(", "))
    df.Year = df.Year.astype(int)

    df_metrics = _process_metrics(df)
    df_model = df[df.topic.apply(lambda x: "Model" in x)].reset_index(drop=True)
    df_model = _process_model(df_model)
    df_model = _process_input_output(df_model)

    print(f"papers -- total: {df.shape[0]}")
    print(f"papers -- new generative method : {df_model.shape[0]}")
    return df, df_metrics, df_model



"""
Matplotlib Blocks
"""
def _bar(
    counts,
    kind="bar",
    x_label="x label",
    y_label="y label",
    rot=0,
    ax=None,
    colors=None,
):
    ax = counts.plot(
        kind=kind,
        # figsize=(10,5),
        color=DEFAULT_PALETTE if not colors else colors,
        edgecolor="white",
        linewidth=0.4,
        rot=rot,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if kind == "bar":
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.grid(axis="x", alpha=0.25)
    return ax


def _stacked_bar(
    counts,
    kind="bar",
    x_label="x label",
    y_label="y label",
    rot=0,
    outside_legend=False,
    ax=None,
    colors=None,
):

    ax = counts.plot(
        kind=kind,
        stacked=True,
        # figsize=(10,5),
        color=DEFAULT_PALETTE if not colors else colors,
        edgecolor="white",
        linewidth=0.4,
        rot=rot,
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if outside_legend:
        ax.legend(
            bbox_to_anchor=(1.02,1),
            loc="upper left",
            # frameon=False,
            title=None
        )
    else:
        ax.legend(
            # bbox_to_anchor=(1.02,1),
            # loc="upper left",
            # frameon=False,
            title=None
        )

    if kind == "bar":
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.grid(axis="x", alpha=0.25)
    return ax


def add_count_to_legend(counts):
    totals = counts.sum(axis=0)
    sorted_counts = (
        totals
        .sort_values(ascending=False)
        .index
    )
    counts = counts[sorted_counts]
    cols = counts.columns.to_list()
    counts.columns = [
        f"{publisher} ({int(totals[publisher])})"
        for publisher in cols
    ]
    return counts



"""
Model Figures
"""
MODEL_COLOR_MAP = {
    'Autoregressive': '#66c2a5',
    'Diffusion': '#fc8d62',
    'Discriminative': '#8da0cb',
    'GAN': '#e78ac3',
    'Hybrid: Discriminative': '#a6d854',
    'Hybrid: Generative': '#ffd92f',
    'VAE': '#e5c494',
}

def plot_model_per_year(
    df_model,
    savepath="data/results/figures/FIGURE_04.png",
    dpi=1_000,
):
    counts = df_model.groupby(['Year', 'Model']).size().unstack(fill_value=0).sort_index()
    counts = add_count_to_legend(counts)

    _stacked_bar(
        counts,
        kind="bar",
        x_label="Year",
        y_label="# papers",
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )



"""
Pre/Post Processing Figures
"""
def _split_levels(x):
    if ':' in x:
        l1, l2 = x.split(':', 1)
        return pd.Series([l1.strip(), l2.strip()])
    else:
        return pd.Series([x.strip(), None])


def _extract_preprocessing_data(df_model):
    col = "Pre processing outside generative model"
    df_model[f"M_{col}"] = df_model[col].apply(lambda x: x.split(", "))

    exploded = df_model[f"M_{col}"].explode()
    exploded = exploded.apply(_split_levels)
    exploded.columns = ['Level1', 'Level2']
    exploded = exploded[exploded["Level1"] != "Absent"]

    counts = exploded.groupby(['Level1', 'Level2']).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    counts = counts.loc[totals.sort_values(ascending=True).index]
    counts = add_count_to_legend(counts)
    return counts


def _extract_postprocessing_data(df_model):
    df_model["postprocessing"] = df_model["Post processing outside generative model"].apply(lambda x: x.split(", "))
    counts = df_model["postprocessing"].explode().value_counts(ascending=True)
    counts = counts.drop("Absent")
    return counts


def plot_pre_post_processing(
    df_model,
    savepath="data/results/figures/FIGURE_05.png",
    dpi=1_000,
):

    pre_counts = _extract_preprocessing_data(df_model)
    post_counts = _extract_postprocessing_data(df_model)
    rot = 15

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    _stacked_bar(
        pre_counts,
        kind="bar",
        x_label="",
        y_label="# papers",
        rot=rot,
        outside_legend=False,
        # outside_legend=True,
        ax=ax[0],
    )

    _bar(
        post_counts,
        kind="bar",
        x_label="",
        y_label="# papers",
        rot=rot,
        ax=ax[1],
    )

    ax[0].set_title("(a) Pre-processing", size=10)
    ax[1].set_title("(b) Post-processing", size=10)

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )

    

"""
Input/Output: Modality
"""
def plot_input_output_modality(
    df_model,
    savepath="data/results/figures/FIGURE_07.png",
    dpi=1_000,
):

    output_counts = df_model["output_modality"].explode().value_counts(ascending=True)
    input_counts = df_model["input_modality"].explode().value_counts(ascending=True)

    input_colors = [
        MODALITY_COLORS[m]
        for m in input_counts.index
    ]

    output_colors = [
        MODALITY_COLORS[m]
        for m in output_counts.index
    ]

    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 4))
    _bar(
        input_counts,
        colors=input_colors,
        kind="barh",
        x_label="# papers",
        y_label="",
        rot=0,
        ax=ax[0],
    )
    
    _bar(
        output_counts,
        colors=output_colors,
        kind="barh",
        x_label="# papers",
        y_label="",
        rot=0,
        ax=ax[1],
    )

    ax[0].set_title("(a) Input Modality", fontsize=10)
    ax[1].set_title("(b) Output Modality", fontsize=10)

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )



"""
Input Content
"""
EXPLICIT_MODALITIES = [
    # floorplans
    "floorplan",
    "partial floorplan",
    # space elements
    "space geometry",
    "floorplan contour",
    # building elements
    "walls/columns",
    "entrance door",
    "interior doors",
    "windows",
]


def _classify_input_modality(modality):
    if modality in EXPLICIT_MODALITIES:
        return "geometric"
    return "non-geometric"


def _get_explicit_intermediate_distinction(df_model):
    df = pd.DataFrame(df_model["input_content"].explode())
    df["explicit/intermediate"] = df["input_content"].apply(_classify_input_modality)
    return df


def plot_input_content(
    df_model,
    # savepath="data/results/figures/LEGACY_01.png",
    # dpi=1_000,
):
    df = _get_explicit_intermediate_distinction(df_model)
    counts = df.groupby(['explicit/intermediate', 'input_content']).size().unstack(fill_value=0)
    counts = add_count_to_legend(counts)

    _stacked_bar(
        counts,
        kind="bar",
        x_label="Content Type",
        y_label="# papers",
        outside_legend=True,
    )

    # plt.savefig(
    #     savepath,
    #     dpi=dpi,
    #     bbox_inches='tight'
    # )


def _double_legend(ax, category_type):
    handles, labels = ax.get_legend_handles_labels()

    grouped_handles = []
    grouped_labels = []

    for group in ["geometric", "non-geometric"]:
        grouped_handles.append(Patch(facecolor="none", edgecolor="none"))
        grouped_labels.append(group)

        for h, label in zip(handles, labels):
            raw_label = label.rsplit(" (", 1)[0]
            if category_type.get(raw_label) == group:
                grouped_handles.append(h)
                grouped_labels.append(label)

    legend = ax.legend(
        grouped_handles,
        grouped_labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        frameon=True,
    )

    # Make section headings visually distinct
    for text in legend.get_texts():
        if text.get_text() in ["geometric", "non-geometric"]:
            text.set_weight("bold")


def plot_input_content(df_model, ax=None):
    df = _get_explicit_intermediate_distinction(df_model)
    # Keep category membership before labels get counts appended
    category_type = (
        df.drop_duplicates("input_content")
          .set_index("input_content")["explicit/intermediate"]
          .to_dict()
    )

    counts = (
        df.groupby(["explicit/intermediate", "input_content"])
          .size()
          .unstack(fill_value=0)
    )
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    counts = add_count_to_legend(counts)

    ax = _stacked_bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="Input Content",
        outside_legend=True,
        ax=ax,
        colors=tab20_colors,
    )

    _double_legend(ax, category_type)
                   


"""
Output Content
"""
def plot_output_content(
    df_model,
    savepath="data/results/figures/FIGURE_09.png",
    dpi=1_000,
):
    counts = df_model["output_content"].explode().value_counts(ascending=True)

    _bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="Output content",
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )



"""
Datasets
"""
def _merge_datasets(x):
    groups = ["LIFULL", "RPLAN", "GeLayout", "MSD", "Tell2Design"]
    return x if x in groups else "other dataset"


def plot_dataset_per_year(
    df,
    savepath="data/results/figures/FIGURE_06.png",
    dpi=1_000,
):

    df["dataset"] = df["Dataset"].apply(lambda x: x.split(", "))

    dataset_exploded = df.explode("dataset")
    dataset_exploded["dataset_grouped"] = dataset_exploded["dataset"].apply(_merge_datasets)

    counts = (
        dataset_exploded
        .groupby(['Year', 'dataset_grouped'])
        .size()
        .unstack(fill_value=0)
    )
    y_lim = counts.values.max()*2

    counts = add_count_to_legend(counts)

    _stacked_bar(
        counts,
        kind="bar",
        x_label="Year",
        y_label="# papers",
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )


"""
Generalizability
"""
def plot_generalizability(
    df,
    savepath="data/results/figures/FIGURE_15.png",
    dpi=1_000,
):

    col = "Generalizability"
    df[f"M_{col}"] = df[col].apply(lambda x: x.split(", "))

    exploded = df[f"M_{col}"].explode()
    exploded = exploded.apply(_split_levels)
    exploded.columns = ['Level1', 'Level2']

    exploded = exploded[exploded["Level1"] != "Absent"]
    counts = exploded.groupby(['Level1', 'Level2']).size().unstack(fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=True).index]

    counts = add_count_to_legend(counts)

    _stacked_bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="Generalizability/Memorization Technique",
        outside_legend=True,
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )


"""
User Studies
"""
def plot_user_studies(
    df,
    savepath="data/results/figures/FIGURE_14.png",
    dpi=1_000,
):

    df["user_studies"] = df["User Studies"].apply(lambda x: x.split(", "))
    counts = df["user_studies"].explode().value_counts(ascending=True)
    counts = counts.drop("Absent")
    
    _bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="User study type",
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )


"""
Metrics: Individual
"""
def _get_s_full(df):
    series = df.apply(lambda row: set.union(*[set(x) for x in row.values]), axis=1)
    series = series.apply(lambda x: x - {'Absent'} if x != {'Absent'} else x)
    return series


def _get_unique_metrics(s_full):

    unique_metrics = list(set.union(*s_full.values))
    unique_metrics.remove("Absent")
    
    counts = []
    for metric in unique_metrics:
        count = s_full.apply(lambda x: metric in x).sum()
        counts.append(count)

    df = pd.DataFrame({
        "metric": unique_metrics,
        "counts": counts
    })

    return df


def plot_num_papers_per_metric(
    df_metrics,
    savepath="data/results/figures/FIGURE_12.png",
    dpi=1_000,
):

    s_full = _get_s_full(df_metrics)
    df_unique = _get_unique_metrics(s_full).set_index("metric")

    df_single = df_unique[df_unique.counts == 1]
    num_singles = df_single.shape[0]
    # s_single = set(df_single.index)
    # num_singles = s_full.apply(lambda x: 1 if len(x.intersection(s_single)) > 0 else 0).sum()

    df_multiple = df_unique[df_unique.counts > 1]
    df_multiple.loc["Other metric", "counts"] = num_singles

    counts = df_multiple.sort_values(by="counts", ascending=True)
    colors = [MODALITY_METRIC_COLORS[x.split(": ")[0]] for x in counts.index]
    counts.index = [i.split(": ")[-1] for i in counts.index]
    counts = counts["counts"]

    legend_handles = [
        Patch(
            facecolor=color,
            label=category
        )
        for category, color in MODALITY_METRIC_COLORS.items()
    ]

    print(f"num of unique metrics: {df_unique.shape[0]}")
    print(f"num of one-time-use metrics: {num_singles}")
    print(f"num of reusable metrics: {df_multiple.shape[0] - 1}")

    ax = counts.plot(
        kind="barh",
        color=colors,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    ax.legend(
        handles=legend_handles,
        title="Metric type",
        loc="lower right",
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )


"""
Metrics: Grouped -- preprocessing
"""
def _process_list(lst):
    lst = [item.split(": ")[0] for item in lst]
    return set(lst)


def get_df_repr(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(_process_list)
    return df


def get_matrix(df_repr):
    s_similarity = df_repr["Similarity"].explode().value_counts(ascending=True)
    s_instruction = df_repr["Instruction-Following"].explode().value_counts(ascending=True)
    s_normative = df_repr["Normative Compliance"].explode().value_counts(ascending=True)
    s_intrinsic = df_repr["Intrinsic"].explode().value_counts(ascending=True)

    s_similarity.name = "Similarity"
    s_instruction.name = "Instruction-Following"
    s_normative.name = "Normative Compliance"
    s_intrinsic.name = "Intrinsic"

    matrix = pd.concat(
        [
            s_similarity,
            s_instruction, 
            s_normative,
            s_intrinsic,
        ], 
        axis=1
    )

    return matrix


def get_s_repr(df):
    series = df.apply(lambda row: set.union(*row.values), axis=1)
    series = series.apply(lambda x: x - {'Absent'} if x != {'Absent'} else x)
    return series


def get_metric_matrix(df_metrics):
    df_repr = get_df_repr(df_metrics)
    matrix = get_matrix(df_repr)
    s_repr = get_s_repr(df_repr)
    return df_repr, s_repr, matrix



"""
Metrics: Grouped -- figures
"""
def plot_stacked_bars(
    df_metrics,
    savepath="data/results/figures/FIGURE_13.png",
    dpi=1_000,
):
    _, _, matrix = get_metric_matrix(df_metrics)

    # Replace NaN with 0 for plotting
    matrix_ = matrix.copy()
    matrix_ = matrix_.drop("Absent").fillna(0).T

    # Sort columns by total
    totals = matrix_.sum(axis=1)
    matrix_ = matrix_.loc[totals.sort_values(ascending=True).index]

    counts = add_count_to_legend(matrix_)

    _stacked_bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="Metric Group",
    )

    plt.savefig(
        savepath,
        dpi=dpi,
        bbox_inches='tight'
    )
    