import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle



"""
LEGACY /// Sankey Block
"""
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
LEGACY /// Input: Modality/Content
"""
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
Input: Modality/Content
"""
def plot_sankey_3levels_from_mapping(
    df_model,
    level1_col="level1",
    level2_col="level2",
    value_col="count",
    level1_order=None,
    level2_order=None,
    level3_order=None,
    title="Modality → Content → Content Group",
    figsize=(18, 12),
    node_width=0.025,
    node_gap=0.018,
    column_x=(0.05, 0.50, 0.95),
    link_alpha=0.25,
    node_alpha=0.85,
    font_size=11,
    save_path="data/results/figures/FIGURE_08.png",
    dpi=1_000,
):
    """
    Draw a three-level Sankey diagram with non-overlapping nodes.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain level1, level2, and count columns.

    level2_to_level3 : dict
        Mapping from every Level-2 category to a Level-3 category.

    level1_order, level2_order, level3_order : list, optional
        Explicit top-to-bottom node order for each level.

    save_path : str, optional
        Example: "sankey.svg", "sankey.pdf", or "sankey.png".
    """

    # prepare data
    _get_csv_modality_content(df_model)
    df = pd.read_csv("data/results/input_modality_content.csv")
    df = df[["level1", "level2", "count"]].copy()

    level2_to_level3 = {
        "space adjacencies/attributes": "non-geometric",
        "floorplan description": "non-geometric",
        "space attributes": "non-geometric",
        "floorplan attributes": "non-geometric",
        "activity map": "non-geometric",


        "floorplan contour": "geometric",
        "entrance door": "geometric",
        "space geometry": "geometric",
        "walls/columns": "geometric",
        "floorplan": "geometric",
        "interior doors": "geometric",
        "partial floorplan": "geometric",
        "windows": "geometric",
    }
    
    # ============================================================
    # 1. Validate and prepare data
    # ============================================================
    required = {level1_col, level2_col, value_col}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = df[[level1_col, level2_col, value_col]].copy()

    data = data.dropna(
        subset=[level1_col, level2_col, value_col]
    )

    data[level1_col] = (
        data[level1_col]
        .astype(str)
        .str.strip()
    )

    data[level2_col] = (
        data[level2_col]
        .astype(str)
        .str.strip()
    )

    data[value_col] = pd.to_numeric(
        data[value_col],
        errors="raise",
    )

    if (data[value_col] < 0).any():
        raise ValueError("Counts cannot be negative.")

    mapping = {
        str(level2).strip(): str(level3).strip()
        for level2, level3 in level2_to_level3.items()
    }

    data["level3"] = data[level2_col].map(mapping)

    unmapped = sorted(
        data.loc[data["level3"].isna(), level2_col]
        .unique()
    )

    if unmapped:
        raise ValueError(
            "Missing Level-2 categories in mapping:\n"
            + "\n".join(f"- {item}" for item in unmapped)
        )

    # ============================================================
    # 2. Aggregate links
    # ============================================================
    links_12 = (
        data.groupby(
            [level1_col, level2_col],
            as_index=False,
        )[value_col]
        .sum()
        .rename(
            columns={
                level1_col: "source",
                level2_col: "target",
                value_col: "value",
            }
        )
    )

    links_23 = (
        data.groupby(
            [level2_col, "level3"],
            as_index=False,
        )[value_col]
        .sum()
        .rename(
            columns={
                level2_col: "source",
                "level3": "target",
                value_col: "value",
            }
        )
    )

    # ============================================================
    # 3. Determine node order
    # ============================================================
    def resolve_order(existing, specified=None):
        existing = list(pd.Index(existing).unique())

        if specified is None:
            return existing

        specified = [str(item).strip() for item in specified]

        ordered = [
            item for item in specified
            if item in existing
        ]

        ordered.extend(
            item for item in existing
            if item not in ordered
        )

        return ordered

    level1_nodes = resolve_order(
        links_12["source"],
        level1_order,
    )

    level3_nodes = resolve_order(
        links_23["target"],
        level3_order,
    )

    level3_position = {
        node: i
        for i, node in enumerate(level3_nodes)
    }

    level2_group = (
        links_23
        .set_index("source")["target"]
        .to_dict()
    )

    raw_level2_nodes = list(
        pd.Index(links_12["target"]).unique()
    )

    if level2_order is None:
        # Group Level-2 nodes by Level-3 destination.
        level2_nodes = sorted(
            raw_level2_nodes,
            key=lambda node: (
                level3_position[level2_group[node]],
                node,
            ),
        )
    else:
        level2_nodes = resolve_order(
            raw_level2_nodes,
            level2_order,
        )

    # ============================================================
    # 4. Calculate node totals
    # ============================================================
    level1_counts = (
        links_12.groupby("source")["value"]
        .sum()
        .to_dict()
    )

    level2_counts = (
        links_12.groupby("target")["value"]
        .sum()
        .to_dict()
    )

    level3_counts = (
        links_23.groupby("target")["value"]
        .sum()
        .to_dict()
    )

    # ============================================================
    # 5. Compute non-overlapping vertical node positions
    # ============================================================
    def calculate_node_boxes(
        nodes,
        counts,
        gap,
        bottom=0.04,
        top=0.96,
    ):
        """
        Return {node: (y_bottom, y_top)}.

        Heights are proportional to counts. Gaps are explicitly
        reserved, so nodes cannot overlap.
        """

        if not nodes:
            return {}

        available_height = top - bottom
        total_gap = gap * max(len(nodes) - 1, 0)
        flow_height = available_height - total_gap

        if flow_height <= 0:
            raise ValueError(
                "node_gap is too large for the number of nodes."
            )

        weights = np.array(
            [float(counts[node]) for node in nodes],
            dtype=float,
        )

        total_weight = weights.sum()

        if total_weight <= 0:
            weights = np.ones(len(nodes))
            total_weight = len(nodes)

        heights = flow_height * weights / total_weight

        boxes = {}
        cursor = top

        # Draw in the supplied top-to-bottom order.
        for node, height in zip(nodes, heights):
            y_top = cursor
            y_bottom = y_top - height

            boxes[node] = (y_bottom, y_top)

            cursor = y_bottom - gap

        return boxes

    boxes_1 = calculate_node_boxes(
        level1_nodes,
        level1_counts,
        node_gap,
    )

    boxes_2 = calculate_node_boxes(
        level2_nodes,
        level2_counts,
        node_gap,
    )

    boxes_3 = calculate_node_boxes(
        level3_nodes,
        level3_counts,
        node_gap,
    )

    # ============================================================
    # 6. Colours
    # ============================================================
    cmap_1 = plt.colormaps["tab20"]
    cmap_2 = plt.colormaps["Set3"]
    cmap_3 = plt.colormaps["Pastel1"]

    colors_1 = {
        node: cmap_1(i % cmap_1.N)
        for i, node in enumerate(level1_nodes)
    }

    colors_2 = {
        node: cmap_2(i % cmap_2.N)
        for i, node in enumerate(level2_nodes)
    }

    colors_3 = {
        node: cmap_3(i % cmap_3.N)
        for i, node in enumerate(level3_nodes)
    }

    # ============================================================
    # 7. Create figure
    # ============================================================
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x1, x2, x3 = column_x

    # ============================================================
    # 8. Ribbon-drawing helper
    # ============================================================
    def draw_ribbon(
        x_start,
        x_end,
        y0_bottom,
        y0_top,
        y1_bottom,
        y1_top,
        color,
        alpha,
    ):
        """
        Draw a smooth filled ribbon between two vertical intervals.
        """

        curvature = 0.45 * (x_end - x_start)

        vertices = [
            (x_start, y0_bottom),

            (x_start + curvature, y0_bottom),
            (x_end - curvature, y1_bottom),
            (x_end, y1_bottom),

            (x_end, y1_top),

            (x_end - curvature, y1_top),
            (x_start + curvature, y0_top),
            (x_start, y0_top),

            (x_start, y0_bottom),
        ]

        codes = [
            Path.MOVETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.LINETO,
            Path.CURVE4,
            Path.CURVE4,
            Path.CURVE4,
            Path.CLOSEPOLY,
        ]

        patch = PathPatch(
            Path(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
            zorder=1,
        )

        ax.add_patch(patch)

    # ============================================================
    # 9. Track consumed portions of each node
    # ============================================================
    outgoing_cursor_1 = {
        node: boxes_1[node][1]
        for node in level1_nodes
    }

    incoming_cursor_2 = {
        node: boxes_2[node][1]
        for node in level2_nodes
    }

    outgoing_cursor_2 = {
        node: boxes_2[node][1]
        for node in level2_nodes
    }

    incoming_cursor_3 = {
        node: boxes_3[node][1]
        for node in level3_nodes
    }

    def node_scale(boxes, counts, node):
        bottom, top = boxes[node]
        return (top - bottom) / counts[node]

    # Sort links according to node order for stable stacking.
    order_1 = {
        node: i for i, node in enumerate(level1_nodes)
    }

    order_2 = {
        node: i for i, node in enumerate(level2_nodes)
    }

    order_3 = {
        node: i for i, node in enumerate(level3_nodes)
    }

    links_12 = (
        links_12.assign(
            source_order=links_12["source"].map(order_1),
            target_order=links_12["target"].map(order_2),
        )
        .sort_values(["source_order", "target_order"])
    )

    links_23 = (
        links_23.assign(
            source_order=links_23["source"].map(order_2),
            target_order=links_23["target"].map(order_3),
        )
        .sort_values(["source_order", "target_order"])
    )

    # ============================================================
    # 10. Draw Level 1 → Level 2 links
    # ============================================================
    for row in links_12.itertuples(index=False):
        source = row.source
        target = row.target
        value = row.value

        source_height = (
            value * node_scale(
                boxes_1,
                level1_counts,
                source,
            )
        )

        target_height = (
            value * node_scale(
                boxes_2,
                level2_counts,
                target,
            )
        )

        source_top = outgoing_cursor_1[source]
        source_bottom = source_top - source_height

        target_top = incoming_cursor_2[target]
        target_bottom = target_top - target_height

        draw_ribbon(
            x_start=x1 + node_width / 2,
            x_end=x2 - node_width / 2,
            y0_bottom=source_bottom,
            y0_top=source_top,
            y1_bottom=target_bottom,
            y1_top=target_top,
            color=colors_1[source],
            alpha=link_alpha,
        )

        outgoing_cursor_1[source] = source_bottom
        incoming_cursor_2[target] = target_bottom

    # ============================================================
    # 11. Draw Level 2 → Level 3 links
    # ============================================================
    for row in links_23.itertuples(index=False):
        source = row.source
        target = row.target
        value = row.value

        source_height = (
            value * node_scale(
                boxes_2,
                level2_counts,
                source,
            )
        )

        target_height = (
            value * node_scale(
                boxes_3,
                level3_counts,
                target,
            )
        )

        source_top = outgoing_cursor_2[source]
        source_bottom = source_top - source_height

        target_top = incoming_cursor_3[target]
        target_bottom = target_top - target_height

        draw_ribbon(
            x_start=x2 + node_width / 2,
            x_end=x3 - node_width / 2,
            y0_bottom=source_bottom,
            y0_top=source_top,
            y1_bottom=target_bottom,
            y1_top=target_top,
            color=colors_2[source],
            alpha=link_alpha,
        )

        outgoing_cursor_2[source] = source_bottom
        incoming_cursor_3[target] = target_bottom

    # ============================================================
    # 12. Draw nodes and labels
    # ============================================================
    def draw_nodes(
        x,
        nodes,
        boxes,
        counts,
        colors,
        label_side="right",
    ):
        for node in nodes:
            y_bottom, y_top = boxes[node]

            rectangle = Rectangle(
                (x - node_width / 2, y_bottom),
                node_width,
                y_top - y_bottom,
                facecolor=colors[node],
                edgecolor=(0, 0, 0, 0.25),
                linewidth=0.7,
                alpha=node_alpha,
                zorder=3,
            )

            ax.add_patch(rectangle)

            y_center = (y_bottom + y_top) / 2

            if label_side == "right":
                text_x = x + node_width / 2 + 0.008
                horizontal_alignment = "left"
            else:
                text_x = x - node_width / 2 - 0.008
                horizontal_alignment = "right"

            ax.text(
                text_x,
                y_center,
                f"{node} ({int(counts[node])})",
                ha=horizontal_alignment,
                va="center",
                fontsize=font_size,
                zorder=4,
            )

    draw_nodes(
        x=x1,
        nodes=level1_nodes,
        boxes=boxes_1,
        counts=level1_counts,
        colors=colors_1,
        label_side="right",
    )

    draw_nodes(
        x=x2,
        nodes=level2_nodes,
        boxes=boxes_2,
        counts=level2_counts,
        colors=colors_2,
        label_side="right",
    )

    draw_nodes(
        x=x3,
        nodes=level3_nodes,
        boxes=boxes_3,
        counts=level3_counts,
        colors=colors_3,
        label_side="left",
    )

    # ============================================================
    # 13. Column headings
    # ============================================================
    ax.text(
        x1,
        0.995,
        # "Level 1",
        "Modality",
        ha="center",
        va="bottom",
        fontsize=font_size + 2,
        fontweight="bold",
    )

    ax.text(
        x2,
        0.995,
        # "Level 2",
        "Content",
        ha="center",
        va="bottom",
        fontsize=font_size + 2,
        fontweight="bold",
    )

    ax.text(
        x3,
        0.995,
        # "Level 3",
        "Content Group",
        ha="center",
        va="bottom",
        fontsize=font_size + 2,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=font_size + 4,
        pad=20,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )



"""
LEGACY /// Input/Output: Information Gain
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
Input/Output: Information Gain
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _replace_notation_input(name):
    dct = {
        "F": "0",
        "A": "NG",
        "B": "B",
        "C": "C",
        "D": "BC",
        "E": "S",
    }

    letter = name[0]
    return dct[letter] + name[1:]


def _replace_notation_output(name):

    dct = {
        "A": "B",
        "B": "S",
        "C": "SB1",
        "D": "SB2",
        "E": "SB3",
        "F": "SB4+",
    }

    letter = name[0]
    return dct[letter] + name[1:]


def _modify_cooc_notation(cooc):
    cooc = cooc.copy()
    cooc["input_content_grouped"] = (
            cooc["input_content_grouped"].apply(_replace_notation_input)
        )
    cooc["output_content_grouped"] = (
        cooc["output_content_grouped"].apply(_replace_notation_output)
    )
    return cooc


def coocurence_heatmap_marginal_distributions(
    df_model,
    savepath="data/results/figures/FIGURE_10.png",
    dpi=1_000,
):

    _, cooc = _get_input_output_groups(df_model)

    # ------------------------------------------------------------
    # 0. Modify output notation A-F -> 1-6
    # ------------------------------------------------------------
    cooc = _modify_cooc_notation(cooc)

    title_fontsize = 14
    fontsize = 12
    legend_fontsize = 11
    legend_title_fontsize = 14

    # ------------------------------------------------------------
    # 1. Validate data
    # ------------------------------------------------------------
    df = cooc

    required_columns = {
        "input_content_grouped",
        "output_content_grouped",
        "Count",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # ------------------------------------------------------------
    # 2. Category order
    # ------------------------------------------------------------
    input_order = [
        "0: empty condition",
        "NG: non-geometric",
        "B: building elements (+ non-geometric)",
        "C: floorplan contour (+ non-geometric)",
        "BC: floorplan contour + building elements (+ non-geometric)",
        "S: floorplan/space geometry (+ any other)",
    ]

    output_order = [
        "S: space geometry",
        "B: walls/columns (+ other building elements)",
        "SB1: space geometry + building elements x1",
        "SB2: space geometry + building elements x2",
        "SB3: space geometry + building elements x3",
        "SB4+: space geometry + building elements x4+",
    ]

    # ------------------------------------------------------------
    # 3. Convert long-form data to matrix
    # ------------------------------------------------------------
    heatmap_data = (
        df.pivot_table(
            index="input_content_grouped",
            columns="output_content_grouped",
            values="Count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(
            index=input_order,
            columns=output_order,
            fill_value=0,
        )
    )

    # ------------------------------------------------------------
    # 4. Short heatmap labels
    # ------------------------------------------------------------
    input_labels = ["0", "NG", "B", "C", "BC", "S"]
    output_labels = ["S", "B", "SB1", "SB2", "SB3", "SB4+"]

    values = heatmap_data.to_numpy()

    # ------------------------------------------------------------
    # 5. Marginal totals
    # ------------------------------------------------------------
    output_marginal = values.sum(axis=0)
    input_marginal = values.sum(axis=1)

    # ------------------------------------------------------------
    # 6. Figure layout
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9))

    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=(5, 1.4, 5),
        height_ratios=(1.4, 3), # !!!
        hspace=0.05, # !!!
        wspace=0.12,
    )

    ax_histx = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    # Legend spans full figure height
    ax_legend = fig.add_subplot(gs[:, 2])

    # ------------------------------------------------------------
    # 7. Main heatmap
    # ------------------------------------------------------------
    image = ax.imshow(
        values,
        aspect=1,
        interpolation="none",
        cmap="Blues",
    )

    ax.set_xticks(np.arange(len(output_labels)))
    ax.set_yticks(np.arange(len(input_labels)))

    ax.set_xticklabels(
        output_labels,
        fontsize=10,
    )

    ax.set_yticklabels(
        input_labels,
        fontsize=10,
    )

    ax.set_xlabel(
        "Output richness",
        fontsize=fontsize,
        labelpad=12,
    )

    ax.set_ylabel(
        "Input richness",
        fontsize=fontsize,
        labelpad=12,
    )

    # ------------------------------------------------------------
    # 8. Annotate heatmap cells
    # ------------------------------------------------------------
    threshold = (
        values.max() / 2
        if values.size and values.max() > 0
        else 0
    )

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):

            value = values[row, col]

            if value == 0:
                continue

            text_color = (
                "white"
                if value > threshold
                else "black"
            )

            ax.text(
                col,
                row,
                f"{value:g}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                fontweight="bold",
            )

    # ------------------------------------------------------------
    # 9. Cell boundaries
    # ------------------------------------------------------------
    ax.set_xticks(
        np.arange(-0.5, len(output_labels), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(input_labels), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        linewidth=1.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    ax.tick_params(
        axis="both",
        which="major",
        length=0,
    )

    # ------------------------------------------------------------
    # 10. Top marginal histogram
    # ------------------------------------------------------------
    x_positions = np.arange(len(output_labels))

    ax_histx.bar(
        x_positions,
        output_marginal,
        width=0.9,
        edgecolor="black",
    )

    ax_histx.set_xlim(
        -0.5,
        len(output_labels) - 0.5,
    )

    ax_histx.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_histx.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_histx.spines["top"].set_visible(False)
    ax_histx.spines["left"].set_visible(False)
    ax_histx.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # 11. Right marginal histogram
    # ------------------------------------------------------------
    y_positions = np.arange(len(input_labels))

    ax_histy.barh(
        y_positions,
        input_marginal,
        height=0.9,
        # height=0.5,
        edgecolor="black",
    )

    # Match heatmap orientation
    ax_histy.set_ylim(ax.get_ylim())

    ax_histy.tick_params(
        axis="y",
        left=False,
        labelleft=False,
    )

    ax_histy.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False,
    )

    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["right"].set_visible(False)
    ax_histy.spines["bottom"].set_visible(False)

    # ------------------------------------------------------------
    # 12. Legend data
    # ------------------------------------------------------------
    input_legend = [
        ("0", "empty condition"),
        ("NG", "non-geometric (graphs, texts, etc)"),
        ("B", "building elements (+ non-geometric)"),
        ("C", "contour (+ non-geometric)"),
        (
            "BC",
            "building elements + contour (+ non-geometric)",
        ),
        (
            "S",
            "space geometry (+ any other)",
        ),
    ]

    output_legend = [
        (
            "S", "space geometry"
        ),
        (
            "B",
            "building elements (walls/columns + other)",
        ),
        (
            "SB1",
            "space geometry + building elements x1",
        ),
        (
            "SB2",
            "space geometry + building elements x2",
        ),
        (
            "SB3",
            "space geometry + building elements x3",
        ),
        (
            "SB4+",
            "space geometry + building elements x4+",
        ),
    ]
    
    # ------------------------------------------------------------
    # 13. Legend formatting
    # ------------------------------------------------------------
    ax_legend.set_xticks([])
    ax_legend.set_yticks([])

    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    for spine in ax_legend.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    # ------------------------------------------------------------
    # Input legend
    # ------------------------------------------------------------
    ax_legend.text(
        0.06,
        0.95,
        "Input richness",
        transform=ax_legend.transAxes,
        fontsize=legend_title_fontsize,
        fontweight="bold",
        va="top",
    )

    y = 0.90

    for code, description in input_legend:

        ax_legend.text(
            0.07,
            y,
            code,
            transform=ax_legend.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
        )

        ax_legend.text(
            0.18,
            y,
            description,
            transform=ax_legend.transAxes,
            fontsize=legend_fontsize,
            va="top",
            linespacing=0.85,
        )

        # Tight spacing between entries
        y -= 0.068

    # ------------------------------------------------------------
    # Divider
    # ------------------------------------------------------------
    divider_y = y + 0.012

    ax_legend.plot(
        [0.06, 0.94],
        [divider_y, divider_y],
        transform=ax_legend.transAxes,
        linewidth=1,
    )

    # ------------------------------------------------------------
    # Output legend
    # ------------------------------------------------------------
    y -= 0.005

    ax_legend.text(
        0.06,
        y,
        "Output richness",
        transform=ax_legend.transAxes,
        fontsize=legend_title_fontsize,
        fontweight="bold",
        va="top",
    )

    y -= 0.050

    for code, description in output_legend:

        ax_legend.text(
            0.07,
            y,
            code,
            transform=ax_legend.transAxes,
            fontsize=fontsize,
            fontweight="bold",
            va="top",
        )

        ax_legend.text(
            0.18,
            y,
            description,
            transform=ax_legend.transAxes,
            fontsize=legend_fontsize,
            va="top",
            linespacing=0.85,
        )

        # Tight spacing between entries
        y -= 0.068

    # ------------------------------------------------------------
    # 14. Title
    # ------------------------------------------------------------
    fig.suptitle(
        "Content: Input → Output",
        fontsize=title_fontsize,
        y=0.97,
    )

    # ------------------------------------------------------------
    # 15. Show
    # ------------------------------------------------------------
    # plt.show()
    fig.savefig(
        savepath,
        dpi=dpi,
        bbox_inches="tight",
    )
    
