import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from copy import deepcopy
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


# def _plot_sankey(
#     df,
#     src_col="level1",
#     trg_col="level2",
#     value_col="count",
#     title_text="Input: Modality → Content",
#     save_path="data/results/input_modality_content.svg",
# ):  
    
#     sources = df[src_col].astype(str)
#     targets = df[trg_col].astype(str)
#     values = df[value_col]

#     nodes = pd.Index(pd.concat([sources, targets]).unique())

#     source_ids = sources.map(nodes.get_loc)
#     target_ids = targets.map(nodes.get_loc)

#     # Node sizes: total in + total out
#     out_counts = df.groupby(src_col)[value_col].sum()
#     in_counts = df.groupby(trg_col)[value_col].sum()

#     node_counts = {
#         node: int(max(out_counts.get(node, 0), in_counts.get(node, 0)))
#         for node in nodes
#     }

#     labels = [
#         f"{node} ({node_counts[node]})"
#         for node in nodes
#     ]

#     fig = go.Figure(data=[go.Sankey(
#         arrangement="snap",
#         node=dict(
#             label=labels,
#             pad=25,
#             thickness=22,
#             line=dict(color="rgba(0,0,0,0.25)", width=0.5),
#         ),
#         link=dict(
#             source=source_ids,
#             target=target_ids,
#             value=values,
#             color="rgba(120,120,120,0.25)",
#         )
#     )])

#     fig.update_layout(
#         title_text=title_text,
#         title_font_size=20,
#         font_size=20,
#         width=1400,
#         height=900,
#         margin=dict(l=40, r=40, t=80, b=40),
#     )

#     fig.write_image(
#         save_path,
#         scale=4,
#     )

#     return fig


def _plot_sankey(
    df,
    src_col="level1",
    trg_col="level2",
    value_col="count",
    title_text="Input: Modality → Content",
    save_path="data/results/input_modality_content.svg",
    src_order=None,
    trg_order=None,
):
    df = df.copy()

    sources = df[src_col].astype(str)
    targets = df[trg_col].astype(str)
    values = df[value_col]

    # Existing unique values
    src_unique = pd.Index(sources.unique())
    trg_unique = pd.Index(targets.unique())

    # If explicit order is given, keep only values that exist,
    # then append any missing values at the end
    if src_order is None:
        src_nodes = list(src_unique)
    else:
        src_order = [str(x) for x in src_order]
        src_nodes = [x for x in src_order if x in src_unique]
        src_nodes += [x for x in src_unique if x not in src_nodes]

    if trg_order is None:
        trg_nodes = list(trg_unique)
    else:
        trg_order = [str(x) for x in trg_order]
        trg_nodes = [x for x in trg_order if x in trg_unique]
        trg_nodes += [x for x in trg_unique if x not in trg_nodes]

    # Final node order: all sources first, all targets second
    nodes = pd.Index(src_nodes + trg_nodes)

    node_to_id = {node: i for i, node in enumerate(nodes)}

    source_ids = sources.map(node_to_id)
    target_ids = targets.map(node_to_id)

    # Node sizes: total in + total out
    out_counts = df.groupby(src_col)[value_col].sum()
    in_counts = df.groupby(trg_col)[value_col].sum()

    node_counts = {
        node: int(max(out_counts.get(node, 0), in_counts.get(node, 0)))
        for node in nodes
    }

    labels = [
        f"{node} ({node_counts[node]})"
        for node in nodes
    ]

    # Explicit coordinates:
    # sources on left, targets on right;
    # y-order follows src_order / trg_order
    n_src = len(src_nodes)
    n_trg = len(trg_nodes)

    x = [0.01] * n_src + [0.99] * n_trg

    y_src = (
        [i / max(n_src - 1, 1) for i in range(n_src)]
        if n_src > 1 else [0.5]
    )
    y_trg = (
        [i / max(n_trg - 1, 1) for i in range(n_trg)]
        if n_trg > 1 else [0.5]
    )

    y = y_src + y_trg

    fig = go.Figure(data=[go.Sankey(
        # arrangement="fixed",  # important: respect x/y
        arrangement="snap",
        node=dict(
            label=labels,
            x=x,
            y=y,
            pad=25,
            thickness=22,
            line=dict(color="rgba(0,0,0,0.25)", width=0.5),
        ),
        link=dict(
            source=source_ids,
            target=target_ids,
            value=values,
            color="rgba(120,120,120,0.25)",
        )
    )])

    fig.update_layout(
        title_text=title_text,
        title_font_size=20,
        font_size=20,
        width=1400,
        height=900,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    if save_path is not None:
        fig.write_image(save_path, scale=4)

    return fig



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


# def plot_model(df_model):    
#     counts = df_model["Model"].value_counts(ascending=True)
    
    
#     counts_col = counts.sort_index().index
#     colors = [MODEL_COLOR_MAP[m] for m in reversed(counts_col)]

#     # colors = [
#     #     MODEL_COLOR_MAP.get(col, "#cccccc")
#     #     for col in counts.index
#     # ]

#     ax = counts.plot(kind="barh", color=colors)
#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#     ax.set_xlabel("# papers")
#     ax.set_ylabel("ML model")


# def plot_discriminative_vs_generative(df_model):
#     counts = df_model.groupby(['Year', 'Discriminative/Generative']).size().unstack(fill_value=0).sort_index()

#     totals = counts.sum(axis=0)
#     sorted_counts = (
#         totals
#         .sort_values(ascending=False)
#         .index
#     )
#     counts = counts[sorted_counts]
#     counts.columns = [
#         f"{publisher} ({totals[publisher]})"
#         for publisher in counts.columns
#     ]

#     # cmap = plt.cm.Set3
#     cmap = plt.cm.tab20b
#     # cmap = plt.cm.terrain
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     ax = counts.plot(
#         kind='bar',
#         stacked=True,
#         rot=0,
#         ylabel="# papers",
#         color=colors,
#     )


def plot_model_per_year(df_model):
    counts = df_model.groupby(['Year', 'Model']).size().unstack(fill_value=0).sort_index()
    
    # colors = [
    #     MODEL_COLOR_MAP.get(col, "#cccccc")
    #     for col in counts.columns
    # ]

    # totals = counts.sum(axis=0)
    # sorted_counts = (
    #     totals
    #     .sort_values(ascending=False)
    #     .index
    # )
    # counts = counts[sorted_counts]
    # cols = counts.columns.to_list()
    # counts.columns = [
    #     f"{publisher} ({totals[publisher]})"
    #     for publisher in cols
    # ]
    counts = add_count_to_legend(counts)

    # ax = counts.plot(
    #     kind='bar',
    #     stacked=True,
    #     rot=0,
    #     ylabel="# papers",
    #     color=colors,
    # )

    _stacked_bar(
        counts,
        kind="bar",
        x_label="Year",
        y_label="# papers",
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


# def two_level_barh(df, col, xlabel=None, ylabel=None):

#     df[f"M_{col}"] = df[col].apply(lambda x: x.split(", "))

#     exploded = df[f"M_{col}"].explode()
#     exploded = exploded.apply(_split_levels)
#     exploded.columns = ['Level1', 'Level2']

#     # exploded['Level2'] = exploded['Level2'].fillna('Absent')
#     exploded = exploded[exploded["Level1"] != "Absent"]
#     # print(exploded)

#     counts = exploded.groupby(['Level1', 'Level2']).size().unstack(fill_value=0)
#     totals = counts.sum(axis=1)
#     counts = counts.loc[totals.sort_values(ascending=True).index]
    
#     # cmap = plt.cm.Set2
#     # num_colors = len(counts.index)
#     # colors = [
#     #     cmap(i / num_colors) for i in range(num_colors)
#     # ]

#     ax = counts.plot(
#         kind='barh',
#         stacked=True,
#         rot=0,
#         # color=colors,
#         # figsize=(2.5, 5)
#     )

#     if xlabel is not None:
#         plt.xlabel(xlabel)
#     if ylabel is not None:
#         plt.ylabel(ylabel)

#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     return ax


# def plot_postprocessing(df_model):
#     df_model["postprocessing"] = df_model["Post processing outside generative model"].apply(lambda x: x.split(", "))
#     counts = df_model["postprocessing"].explode().value_counts(ascending=True)
#     counts = counts.drop("Absent")

#     # cmap = plt.cm.tab20
#     cmap = plt.cm.tab20c
#     # cmap = plt.cm.Set3
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     ax = counts.plot(
#         kind="bar", 
#         xlabel="# papers", 
#         ylabel="(b) Post-processing technique",
#         color=colors,
#         # figsize=(2.5, 5),
#         rot=0,

#     )

#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     return ax


# def plot_pre_post_processing(df_model):

#     fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

#     col = "Pre processing outside generative model"
#     df_model[f"M_{col}"] = df_model[col].apply(lambda x: x.split(", "))

#     exploded = df_model[f"M_{col}"].explode()
#     exploded = exploded.apply(_split_levels)
#     exploded.columns = ['Level1', 'Level2']
#     exploded = exploded[exploded["Level1"] != "Absent"]

#     counts = exploded.groupby(['Level1', 'Level2']).size().unstack(fill_value=0)
#     totals = counts.sum(axis=1)
#     counts = counts.loc[totals.sort_values(ascending=True).index]

#     # cmap = plt.cm.Set2
#     # num_colors = len(counts.index)
#     # colors = [
#     #     cmap(i / num_colors) for i in range(num_colors)
#     # ]

#     counts.plot(
#         kind='bar',
#         stacked=True,
#         rot=45,
#         ax=ax[0]
#     )

#     ax[0].set_ylim(
#         0,
#         10 * 1,
#     )
#     ax[0].set_xlabel("")
#     ax[0].set_ylabel("# papers")
#     ax[0].set_title("(a) Pre-processing")

#     df_model["postprocessing"] = df_model["Post processing outside generative model"].apply(lambda x: x.split(", "))
#     counts = df_model["postprocessing"].explode().value_counts(ascending=True)
#     counts = counts.drop("Absent")

#     # cmap = plt.cm.tab20
#     cmap = plt.cm.tab20c
#     # cmap = plt.cm.Set3
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     counts.plot(
#         kind="bar", 
#         color=colors,
#         rot=45,
#         ax=ax[1]
#     )

#     ax[1].set_ylim(
#         0,
#         10 * 1,
#     )
#     ax[1].set_xlabel("")
#     ax[1].set_ylabel("# papers")
#     ax[1].set_title("(b) Post-processing")


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


def plot_pre_post_processing(df_model):

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

    

"""
Input/Output: Modality
"""
# def plot_input_modality(df_model, ax=None):

#     counts = df_model["input_modality"].explode().value_counts(ascending=True)

#     cmap = plt.cm.tab20
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     counts.plot(
#         kind="barh", 
#         xlabel="# papers", 
#         ylabel="Input modality",
#         color=colors,
#         ax=ax,
#     )


# def plot_output_modality(df_model, ax=None):
#     df_model["output_modality"] = df_model["Neural Output (modality)"].apply(lambda x: x.split(", "))
#     counts = df_model["output_modality"].explode().value_counts(ascending=True)

#     cmap = plt.cm.Set3
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     counts.plot(
#         kind="barh", 
#         xlabel="# papers", 
#         ylabel="Output modality",
#         color=colors,
#         ax=ax,
#     )


# def plot_input_output_modality(df_model):

#     fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 4))
    
#     plot_input_modality(df_model, ax=ax[0])
#     plot_output_modality(df_model, ax=ax[1])

#     ax[0].set_title("(a)", fontsize=11)
#     ax[1].set_title("(b)", fontsize=11)

#     plt.tight_layout()
#     plt.show()


def plot_input_output_modality(df_model):

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

    plt.tight_layout()
    plt.show()



# """
# Input/Output: Content
# """
# def plot_input_content(df_model, ax=None):

#     counts = df_model["input_content"].explode().value_counts(ascending=True)

#     cmap = plt.cm.tab20
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     counts.plot(
#         kind="barh", 
#         xlabel="# papers", 
#         ylabel="Input content",
#         color=colors,
#         ax=ax,
#     )


# def plot_output_content(df_model, ax=None):
#     counts = df_model["output_content"].explode().value_counts(ascending=True)

#     cmap = plt.cm.Set3
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     counts.plot(
#         kind="barh", 
#         xlabel="# papers", 
#         ylabel="Output content",
#         color=colors,
#         ax=ax,
#     )


# def plot_input_output_content(df_model):

#     fig, ax = plt.subplots(1, 2, sharex=True, figsize=(9, 4))
    
#     plot_input_content(df_model, ax=ax[0])
#     plot_output_content(df_model, ax=ax[1])

#     ax[0].set_title("(a)", fontsize=11)
#     ax[1].set_title("(b)", fontsize=11)

#     plt.tight_layout()
#     plt.show()


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


def plot_input_content(df_model):
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


# def plot_output_content(df_model, ax=None):
#     counts = df_model["output_content"].explode().value_counts(ascending=True)

#     _bar(
#         counts,
#         kind="barh",
#         x_label="# papers",
#         y_label="Output content",
#         # rot=0,
#         # ax=None,
#         # colors=None,
#     )

def _double_legend(ax, category_type):
    handles, labels = ax.get_legend_handles_labels()
    # if ax.legend_ is not None:
    #     ax.legend_.remove()

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
def plot_output_content(df_model, ax=None):
    counts = df_model["output_content"].explode().value_counts(ascending=True)

    _bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="Output content",
        # rot=0,
        # ax=None,
        # colors=None,
    )


"""
Input: Modality/Content
"""
# def _process_lvl1(name):
#     dct = {
#         "pixels": "A: pixels",
#         "graphs": "B: graphs",
#         "sequence": "C: sequence",
#         "vector": "D: vector",
#         "polygons": "E: polygons",
#         "wall graphs": "F: wall graphs",
#     }
#     return dct[name]


def _get_csv_modality_content(df_model):
    levels = df_model["input"].explode().str.split(':', n=1, expand=True)
    levels.columns = ['level1', 'level2']

    # Clean whitespace
    levels = levels.apply(lambda col: col.str.strip())

    # Count co-occurrences
    counts = (
        levels
        .groupby(['level1', 'level2'])
        .size()
        .reset_index(name='count')
    )

    # Export
    counts.to_csv("data/results/input_modality_content.csv", index=False)


def plot_sankey_input_modality_content(df_model):
    _get_csv_modality_content(df_model)

    df = pd.read_csv("data/results/input_modality_content.csv")
    df = df[["level1", "level2", "count"]].copy()

    # Optional: remove very small flows
    # df = df[df["count"] >= 2]

    # Sort for more stable/readable layout
    df = df.sort_values(["level1", "count"], ascending=[True, False])
    # df["level1"] = df["level1"].apply(_process_lvl1)

    return _plot_sankey(
        df,
        src_col="level1",
        trg_col="level2",
        value_col="count",
        title_text="Input: Modality → Content",
        save_path="data/results/input_modality_content.svg",
    )



"""
Input/Output: Information Gain
"""
def _aggregate_input_content(s):
    building_elements = {"walls/columns", "entrance door", "interior door", "windows"}

    if not len(s):
        # return "0: empty condition"
        return "F: empty condition"
    elif {"floorplan", "partial floorplan", "space geometry"} & s:
        return "E: floorplan/space geometry (+ any other)"
    elif {"floorplan contour"} & s:
        if building_elements & s:
            return "D: floorplan contour + building elements (+ non-geometric)"
        else:
            return "C: floorplan contour (+ non-geometric)"
    elif building_elements & s:
        return "B: building elements (+ non-geometric)"
    else:
        # print(s)
        return f"A: non-geometric"


# def _aggregate_output_content(s):
#     if "space contour/function" in s:
#         result = "space geometry"
#         length = len(s) - 1
#         if length > 0:
#             if length >= 4:
#                 length = "4+"
#             result += f" + anchors x{length}"

#     else:
#         result = "anchors x1+"
#     return result


def _aggregate_output_content(s):
    if "space contour/function" not in s:
        result = "A: walls/columns (+ other building elements)"
    else:
        result = "space geometry"
        length = len(s) - 1
        if length == 0:
            result = f"B: {result}"
        elif length == 1:
            result = f"C: {result} + building elements x1"
        elif length == 2:
            result = f"D: {result} + building elements x2"
        elif length == 3:
            result = f"E: {result} + building elements x3"
        else:
            result = f"F: {result} + building elements x4+"
    return result



def _get_input_output_groups(df_model):
    df_model = df_model.copy()
    print(f"dropped rows: {df_model[df_model["Conditioning Regimes"] == "either"].shape[0]}")
    df_model = df_model[df_model["Conditioning Regimes"] != "either"]
    df_model['input_content_grouped'] = df_model['input_content'].apply(_aggregate_input_content)
    df_model['output_content_grouped'] = df_model['output_content'].apply(_aggregate_output_content)

    cooc = (
        df_model
        .groupby(['input_content_grouped', 'output_content_grouped'])
        .size()
        .reset_index(name='Count')
    ).sort_values("Count", ascending=False)

    df_model = df_model[["Title", "input_content", "input_content_grouped", "output_content", "output_content_grouped"]]
    return df_model, cooc


def print_information_gain_groups(df_):

    best = df_[(df_.input_content_grouped == "intermediates x1") & (df_.output_content_grouped == "space geometry + anchors x4+")]
    print("highest information gain:")
    for title in best.Title:
        print("-", title)
    print()

    worst = df_[df_.input_content_grouped == "complete floorplan"]
    print("zero information gain:")
    for title in worst.Title:
        print("-", title)
    print()
    print()


# def plot_subgroups(df_):
#     fig, ax = plt.subplots(2, 3, sharex=True)

#     df_[df_.input_content_grouped == "intermediates x1"].input_content.value_counts().plot(kind="barh", ax=ax[0][0])
#     df_[df_.input_content_grouped == "intermediates x2+"].input_content.value_counts().plot(kind="barh", ax=ax[0][1])
#     df_[df_.input_content_grouped == "intermediates + anchors"].input_content.value_counts().plot(kind="barh", ax=ax[0][2])
#     df_[df_.input_content_grouped == "space geometry + other"].input_content.value_counts().plot(kind="barh", ax=ax[1][0])
#     df_[df_.input_content_grouped == "complete floorplan"].input_content.value_counts().plot(kind="barh", ax=ax[1][1])
#     df_[df_.input_content_grouped == "empty condition"].input_content.value_counts().plot(kind="barh", ax=ax[1][2])


# def plot_input_output_groups(df_model):

#     df_, cooc = _get_input_output_groups(df_model)
#     df_.to_csv("data/results/input_output_content_groups_titles.csv", index=False)
#     cooc.to_csv("data/results/input_output_content_groups_cooc.csv", index=False)
#     print_information_gain_groups(df_)

#     sources = cooc["input_content_grouped"].astype(str)
#     targets = cooc["output_content_grouped"].astype(str)
#     values = cooc["Count"]

#     # Create unique node list
#     nodes = pd.Index(pd.concat([sources, targets]).unique())

#     # Map labels to numeric IDs
#     source_ids = sources.map(nodes.get_loc)
#     target_ids = targets.map(nodes.get_loc)

#     fig = go.Figure(data=[go.Sankey(
#         node=dict(
#             label=nodes.tolist(),
#             pad=15,
#             thickness=18,
#         ),
#         link=dict(
#             source=source_ids,
#             target=target_ids,
#             value=values
#         )
#     )])

#     fig.update_layout(
#         title_text="Content: Input → Output",
#         title_font_size=20,
#         width=1200,
#         height=800,
#         font=dict(
#             color="brown",
#             size=20,
#         )
#     )

#     fig.write_image(
#         "data/results/input_output_content_groups.svg",
#         scale=4,
#     )

#     return fig


def plot_sankey_input_output_content(df_model):

    df, cooc = _get_input_output_groups(df_model)
    df.to_csv("data/results/input_output_content_groups_titles.csv", index=False)
    cooc.to_csv("data/results/input_output_content_groups_cooc.csv", index=False)
    # print_information_gain_groups(df)

    # Optional: remove very small flows
    # df = df[df["count"] >= 2]

    # Sort for more stable/readable layout
    cooc = cooc.sort_values(["input_content_grouped", "Count"], ascending=[True, False])

    return _plot_sankey(
        cooc,
        src_col="input_content_grouped",
        trg_col="output_content_grouped",
        value_col="Count",
        title_text="Content: Input → Output",
        save_path="data/results/input_output_content_groups.svg",
    )



"""
Datasets
"""
def _merge_datasets(x):
    groups = ["LIFULL", "RPLAN", "GeLayout", "MSD", "Tell2Design"]
    return x if x in groups else "other dataset"


def plot_dataset_per_year(df):

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

    # totals = counts.sum(axis=0)
    # sorted_counts = (
    #     totals
    #     .sort_values(ascending=False)
    #     .index
    # )
    # counts = counts[sorted_counts]
    # counts.columns = [
    #     f"{publisher} ({totals[publisher]})"
    #     for publisher in counts.columns
    # ]

    counts = add_count_to_legend(counts)

    # # cmap = plt.cm.Set3
    # cmap = plt.cm.tab20c
    # num_colors = len(counts.index)
    # colors = [
    #     cmap(i / num_colors) for i in range(num_colors)
    # ]

    # ax = counts.plot(
    #     kind='bar',
    #     stacked=True,
    #     ylabel='# papers',
    #     color=colors,
    #     # figsize=(12,6)
    # )
    # ax.tick_params(axis='x', rotation=0)
    # # ax.grid(axis='y')
    # ax.set_ylim(
    #     0,
    #     y_lim * 1.1,
    # )

    
    # _stacked_bar_vertical(
    #     counts,
    #     "Year",
    #     "# papers",
    # )

    _stacked_bar(
        counts,
        kind="bar",
        x_label="Year",
        y_label="# papers",
    )



"""
Generalizability
"""
# def generalizability_barh(df, col, xlabel=None, ylabel=None):

#     df[f"M_{col}"] = df[col].apply(lambda x: x.split(", "))

#     exploded = df[f"M_{col}"].explode()
#     exploded = exploded.apply(_split_levels)
#     exploded.columns = ['Level1', 'Level2']

#     # exploded['Level2'] = exploded['Level2'].fillna('Absent')
#     exploded = exploded[exploded["Level1"] != "Absent"]
#     # print(exploded)

#     counts = exploded.groupby(['Level1', 'Level2']).size().unstack(fill_value=0)
#     totals = counts.sum(axis=1)
#     counts = counts.loc[totals.sort_values(ascending=True).index]
    
#     # cmap = plt.cm.Set2
#     # num_colors = len(counts.index)
#     # colors = [
#     #     cmap(i / num_colors) for i in range(num_colors)
#     # ]

#     ax = counts.plot(
#         kind='barh',
#         stacked=True,
#         rot=0,
#         # color=colors,
#         # figsize=(2.5, 5)
#     )
    
#     ax.legend(
#         loc='center left',
#         bbox_to_anchor=(1, 0.5)
#     )
    
#     if xlabel is not None:
#         plt.xlabel(xlabel)
#     if ylabel is not None:
#         plt.ylabel(ylabel)

#     ax.set_xlim(
#         0,
#         9,
#     )

#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#     return ax


def plot_generalizability(df):

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



"""
User Studies
"""
# def plot_user_studies(df):

#     df["user_studies"] = df["User Studies"].apply(lambda x: x.split(", "))
#     counts = df["user_studies"].explode().value_counts(ascending=True)
#     counts = counts.drop("Absent")
    
#     cmap = plt.cm.Set3
#     # cmap = plt.cm.Pastel1
#     # cmap = plt.cm.tab20b
    
#     num_colors = len(counts.index)
#     colors = [
#         cmap(i / num_colors) for i in range(num_colors)
#     ]

#     ax = counts.plot(
#         kind="barh", 
#         xlabel="# papers", 
#         ylabel="User study type",
#         color=colors,
#     )

#     ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_user_studies(df):

    df["user_studies"] = df["User Studies"].apply(lambda x: x.split(", "))
    counts = df["user_studies"].explode().value_counts(ascending=True)
    counts = counts.drop("Absent")
    
    _bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="User study type",
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


# def plot_num_papers_per_metric(df_metrics):

#     s_full = _get_s_full(df_metrics)
#     df_unique = _get_unique_metrics(s_full).set_index("metric")
#     df_multiple = df_unique[df_unique.counts > 1]
#     df_single = df_unique[df_unique.counts == 1]

#     s_single = set(df_single.index)
#     num_singles = s_full.apply(lambda x: 1 if len(x.intersection(s_single)) > 0 else 0).sum()

#     df_multiple.loc["Other metric", "counts"] = num_singles
#     df_multiple.index = [x.split(": ")[-1] for x in df_multiple.index]


#     print(f"num of unique metrics: {df_unique.shape[0]}")
#     print(f"num of one-time-use metrics: {num_singles}")
#     print(f"num of reusable metrics: {df_unique.shape[0] - num_singles}")

#     counts = df_multiple.sort_values(by="counts", ascending=True)

#     # cmap = plt.cm.Set3
#     # # cmap = plt.cm.Pastel1
#     # # cmap = plt.cm.tab20b
    
#     # num_colors = len(counts.index)
#     # colors = [
#     #     cmap(i / num_colors) for i in range(num_colors)
#     # ]

#     counts.plot(
#         kind="barh", 
#         legend=False,
#         xlabel="# papers",
#         ylabel="Metric",
#         # color=colors,
#     )

#     # return df_multiple


def plot_num_papers_per_metric(df_metrics):

    s_full = _get_s_full(df_metrics)
    df_unique = _get_unique_metrics(s_full).set_index("metric")
    df_multiple = df_unique[df_unique.counts > 1]
    df_single = df_unique[df_unique.counts == 1]

    s_single = set(df_single.index)
    num_singles = s_full.apply(lambda x: 1 if len(x.intersection(s_single)) > 0 else 0).sum()
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
    print(f"num of reusable metrics: {df_unique.shape[0] - num_singles}")

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
# def plot_num_papers_per_repr(s_repr):
#     s = s_repr.explode()
#     s = s[s != "Absent"]
#     counts = s.value_counts(ascending=True)

#     color_map = {
#         "Embeddings": "#E15759",
#         "Regions": "#59A14F",
#         "Pixels": "#F28E2B",
#         "Attributes": "#4E79A7",
#         "Graphs":  "#F5B2D9",
#     }
#     colors = [color_map.get(label, "#cccccc") for label in counts.index.astype(str)]

#     fig, ax = plt.subplots()
#     ax.barh(
#         counts.index,
#         counts,
#         color=colors,
#     )

#     # ax.set_xlabel("# papers")
#     # ax.set_ylabel("Metric Group")
#     # # ax.grid(axis="x", linestyle="--", alpha=0.4)


# def num_papers_per_group(df_repr):
#     cols = []
#     counts = []
#     for col in df_repr.columns:
#         count = df_repr[col][df_repr[col] != {'Absent'}].shape[0]
#         cols.append(col)
#         counts.append(count)

#     df = pd.DataFrame({"group": cols, "count": counts})
#     df = df.set_index("group")
#     df = df.sort_values(by="count", ascending=True)
#     # return df

#     color_map = {
#         "Normative Compliance": "#E15759",
#         "Intrinsic": "#59A14F",
#         "Instruction-Following": "#F28E2B",
#         "Similarity": "#4E79A7",
#     }
#     colors = [color_map.get(label, "#cccccc") for label in df.index.astype(str)]
#     # print (colors)
    
#     # fig, ax = plt.subplots(figsize=(6, 4))
#     fig, ax = plt.subplots()

#     ax.barh(
#         df.index,
#         df["count"],
#         color=colors,
#     )

#     ax.set_xlabel("# papers")
#     ax.set_ylabel("Metric Group")
#     # ax.grid(axis="x", linestyle="--", alpha=0.4)


# def plot_heatmap(matrix):
#     plt.figure(figsize=(8, 5))

#     sns.heatmap(
#         # matrix,
#         matrix.drop("Absent"),
#         annot=True,
#         fmt=".0f",
#         cmap="YlGnBu",
#         linewidths=0.5,
#         cbar_kws={"label": "# papers"}
#     )

#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)
#     plt.xlabel("Automatic Function Group")
#     plt.ylabel("Representation")
#     plt.tight_layout()

#     plt.show()


def plot_stacked_bars(df_metrics):
    
    _, _, matrix = get_metric_matrix(df_metrics)

    # Replace NaN with 0 for plotting
    matrix_ = matrix.copy()
    matrix_ = matrix_.drop("Absent").fillna(0).T

    # Sort columns by total
    totals = matrix_.sum(axis=1)
    matrix_ = matrix_.loc[totals.sort_values(ascending=True).index]

    counts = add_count_to_legend(matrix_)

    # ax = matrix_.plot(
    #     kind="barh",
    #     stacked=True,
    #     # figsize=(5, 5),
    #     # width=0.8,
    # )
    
    # ax.set_xlabel("# papers")
    # ax.set_ylabel("Automatic Function Group")

    # ax.legend(
    #     title="Representation",
    #     # bbox_to_anchor=(1.02, 0.5),
    #     # loc="center left"
    # )

    # # plt.xticks(rotation=45)
    # # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.show()

    # _stacked_bar_horisontal(
    #     matrix_,
    #     "Metric Group",
    #     "# papers",
    # )

    _stacked_bar(
        counts,
        kind="barh",
        x_label="# papers",
        y_label="Metric Group",
    )