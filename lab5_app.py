import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Lab 5 – High-Dimensional Data", layout="wide")

# ── Dataset configurations ───────────────────────────────────────────────────
# Each chart reads its columns from these "roles", so swapping the dataset
# (Exercise 3.1) re-runs every visualization without touching the chart code.
DATASETS = {
    "Tips (original)": {
        "file": "tips.csv",
        "cat_a": "day", "cat_b": "sex", "value": "total_bill", "value_agg": "sum",
        "x": "total_bill", "y": "tip", "size": "size", "color": "sex",
        "facet": "day",
        "heat_row": "day", "heat_col": "time", "heat_val": "total_bill",
        "parallel": ["total_bill", "tip", "size"], "parallel_cat": "day",
        "xyz": ["total_bill", "tip", "size"], "log": [],
        "tooltip_extra": ["smoker", "time"],
    },
    "Gapminder 2007 (custom)": {
        "file": "gapminder_2007.csv",
        "cat_a": "continent", "cat_b": "income_level", "value": "pop", "value_agg": "sum",
        "x": "gdpPercap", "y": "lifeExp", "size": "pop", "color": "continent",
        "facet": "continent",
        "heat_row": "continent", "heat_col": "income_level", "heat_val": "lifeExp",
        "parallel": ["gdpPercap", "lifeExp", "pop"], "parallel_cat": "continent",
        "xyz": ["gdpPercap", "lifeExp", "pop"], "log": ["gdpPercap", "pop"],
        "tooltip_extra": ["country"],
    },
}

# Fixed category orders so the charts read in a natural sequence, not alphabetically
ORDERS = {
    "day": ["Thur", "Fri", "Sat", "Sun"],
    "time": ["Lunch", "Dinner"],
    "income_level": ["Low", "Lower-middle", "Upper-middle", "High"],
}

LABELS = {
    "total_bill": "Total bill ($)", "tip": "Tip ($)", "size": "Party size",
    "sex": "Sex", "day": "Day", "time": "Time", "smoker": "Smoker",
    "gdpPercap": "GDP per capita (US$)", "lifeExp": "Life expectancy (years)",
    "pop": "Population", "continent": "Continent", "income_level": "Income level",
    "country": "Country",
}


def label(col):
    return LABELS.get(col, col)


@st.cache_data
def load(file):
    return pd.read_csv(file)


# ── Sidebar controls (Exercises 3.1 and 3.2) ─────────────────────────────────
st.sidebar.header("Dataset")
ds_name = st.sidebar.selectbox("Choose dataset", list(DATASETS))
cfg = DATASETS[ds_name]
df = load(cfg["file"])

numeric_cols = df.select_dtypes("number").columns.tolist()

st.sidebar.header("Customize")
x_col = st.sidebar.selectbox("Scatter X", numeric_cols, index=numeric_cols.index(cfg["x"]))
y_col = st.sidebar.selectbox("Scatter Y", numeric_cols, index=numeric_cols.index(cfg["y"]))
scheme = st.sidebar.selectbox("Colour scheme", ["tableau10", "set2", "dark2", "category10"])
heat_scheme = st.sidebar.selectbox("Heatmap scheme", ["blues", "viridis", "orangered", "greens"])
point_size = st.sidebar.slider("Max point size", 100, 1500, 600, step=100)

st.title("Lab 5: High-Dimensional Data Visualization Techniques")
st.write(f"Dataset: **{ds_name}** — {len(df)} rows, {len(df.columns)} columns")
with st.expander("Preview data"):
    st.dataframe(df.head(20), width="stretch")


def sort_order(col):
    return ORDERS.get(col, sorted(df[col].unique()))


# ── 2.1 Mosaic Plot (Marimekko) ──────────────────────────────────────────────
st.header("2.1 Mosaic Plot (Marimekko)")

a, b, v = cfg["cat_a"], cfg["cat_b"], cfg["value"]
mosaic = df.groupby([a, b], observed=True)[v].sum().reset_index()
col_tot = mosaic.groupby(a)[v].transform("sum")
mosaic["share_in_col"] = mosaic[v] / col_tot

# Column widths are proportional to each category's share of the grand total,
# so area = joint share; that is what distinguishes a mosaic from a stacked bar.
col_share = mosaic.groupby(a)[v].sum().reindex(sort_order(a)).dropna() / mosaic[v].sum()
x_end = col_share.cumsum()
mosaic["x0"] = mosaic[a].map(x_end - col_share)
mosaic["x1"] = mosaic[a].map(x_end)
mosaic["b_rank"] = mosaic[b].map({k: i for i, k in enumerate(sort_order(b))})
mosaic = mosaic.sort_values([a, "b_rank"])
mosaic["y1"] = mosaic.groupby(a)["share_in_col"].cumsum()
mosaic["y0"] = mosaic["y1"] - mosaic["share_in_col"]
mosaic["x_mid"] = (mosaic["x0"] + mosaic["x1"]) / 2

rects = alt.Chart(mosaic).mark_rect(stroke="white", strokeWidth=2).encode(
    x=alt.X("x0:Q", axis=alt.Axis(format="%", title=f"Share of {label(v)} by {label(a)}")),
    x2="x1",
    y=alt.Y("y0:Q", axis=alt.Axis(format="%", title=f"Share within {label(a)}")),
    y2="y1",
    color=alt.Color(f"{b}:N", sort=sort_order(b), title=label(b),
                    scale=alt.Scale(scheme=scheme)),
    tooltip=[
        alt.Tooltip(f"{a}:N", title=label(a)),
        alt.Tooltip(f"{b}:N", title=label(b)),
        alt.Tooltip(f"{v}:Q", title=f"Sum of {label(v)}", format=",.0f"),
        alt.Tooltip("share_in_col:Q", title=f"Share within {label(a)}", format=".1%"),
    ],
)
col_labels = alt.Chart(mosaic.drop_duplicates(a)).mark_text(dy=-8, fontWeight="bold").encode(
    x="x_mid:Q", y=alt.value(0), text=f"{a}:N"
)
st.altair_chart(
    (rects + col_labels).properties(
        height=380, title=f"Mosaic: sum of {label(v)} by {label(a)} and {label(b)}"
    ),
    width="stretch",
)
st.caption(
    "Column width = each group's share of the total; segment height = the split inside "
    "that group. Every rectangle's area is its share of the whole."
)

# ── 2.2 Trellis Displays ─────────────────────────────────────────────────────
st.header("2.2 Trellis Displays (Small Multiples)")

x_scale = alt.Scale(type="log") if x_col in cfg["log"] else alt.Scale(zero=False)
trellis = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X(f"{x_col}:Q", title=label(x_col), scale=x_scale),
    y=alt.Y(f"{y_col}:Q", title=label(y_col), scale=alt.Scale(zero=False)),
    color=alt.Color(f"{cfg['color']}:N", title=label(cfg["color"]),
                    scale=alt.Scale(scheme=scheme)),
    tooltip=[alt.Tooltip(c, title=label(c)) for c in cfg["tooltip_extra"] + [x_col, y_col]],
).properties(width=220, height=200).facet(
    facet=alt.Facet(f"{cfg['facet']}:N", title=label(cfg["facet"]), sort=sort_order(cfg["facet"])),
    columns=4,
).properties(title=f"{label(y_col)} vs {label(x_col)}, one panel per {label(cfg['facet'])}")
st.altair_chart(trellis)
st.caption("Panels share axes, so differences between groups can be compared by eye.")

# ── 2.3 Heatmap ──────────────────────────────────────────────────────────────
st.header("2.3 Heatmap")

hr, hc, hv = cfg["heat_row"], cfg["heat_col"], cfg["heat_val"]
heat_df = df.groupby([hr, hc], observed=True)[hv].agg(["mean", "count"]).reset_index()
heat_mid = (heat_df["mean"].min() + heat_df["mean"].max()) / 2
heat_base = alt.Chart(heat_df).encode(
    x=alt.X(f"{hc}:N", title=label(hc), sort=sort_order(hc), axis=alt.Axis(labelAngle=0)),
    y=alt.Y(f"{hr}:N", title=label(hr), sort=sort_order(hr)),
)
heat = heat_base.mark_rect().encode(
    color=alt.Color("mean:Q", title=f"Mean {label(hv)}",
                    scale=alt.Scale(scheme=heat_scheme)),
    tooltip=[
        alt.Tooltip(f"{hr}:N", title=label(hr)),
        alt.Tooltip(f"{hc}:N", title=label(hc)),
        alt.Tooltip("mean:Q", title=f"Mean {label(hv)}", format=".1f"),
        alt.Tooltip("count:Q", title="Records"),
    ],
)
# White text on the darker half of the colour scale so every value stays readable
heat_text = heat_base.mark_text(fontSize=13).encode(
    text=alt.Text("mean:Q", format=".1f"),
    color=alt.condition(alt.datum.mean > heat_mid, alt.value("white"), alt.value("black")),
)
st.altair_chart(
    (heat + heat_text).properties(
        height=320, title=f"Mean {label(hv)} by {label(hr)} and {label(hc)}"
    ),
    width="stretch",
)
st.caption("Blank cells mean no records exist for that combination.")

# ── 2.4 Multivariate Scatter Plot ────────────────────────────────────────────
st.header("2.4 Multivariate Scatter Plot")

shape_col = cfg["cat_b"]
multi = alt.Chart(df).mark_point(filled=True, opacity=0.75).encode(
    x=alt.X(f"{x_col}:Q", title=label(x_col), scale=x_scale),
    y=alt.Y(f"{y_col}:Q", title=label(y_col), scale=alt.Scale(zero=False)),
    color=alt.Color(f"{cfg['color']}:N", title=label(cfg["color"]),
                    scale=alt.Scale(scheme=scheme)),
    shape=alt.Shape(f"{shape_col}:N", title=label(shape_col), sort=sort_order(shape_col)),
    size=alt.Size(f"{cfg['size']}:Q", title=label(cfg["size"]),
                  scale=alt.Scale(range=[20, point_size])),
    tooltip=[alt.Tooltip(c, title=label(c))
             for c in cfg["tooltip_extra"] + [x_col, y_col, cfg["size"], cfg["color"], shape_col]],
).properties(
    height=450,
    title=f"{label(y_col)} vs {label(x_col)} — colour: {label(cfg['color'])}, "
          f"shape: {label(shape_col)}, size: {label(cfg['size'])}",
).interactive()
st.altair_chart(multi, width="stretch")
st.caption("Five variables in one view: x, y, colour, shape and size. Scroll to zoom, drag to pan.")

# ── 2.5 Parallel Coordinate Plot ─────────────────────────────────────────────
st.header("2.5 Parallel Coordinate Plot")

pc = cfg["parallel_cat"]
pc_levels = sort_order(pc)
pc_codes = df[pc].map({k: i for i, k in enumerate(pc_levels)})
dims = [
    dict(label=label(c), values=np.log10(df[c]) if c in cfg["log"] else df[c])
    for c in cfg["parallel"]
]
for d, c in zip(dims, cfg["parallel"]):
    if c in cfg["log"]:
        d["label"] += " (log10)"
dims.append(dict(label=label(pc), values=pc_codes,
                 tickvals=list(range(len(pc_levels))), ticktext=pc_levels))

palette = px.colors.qualitative.T10
n = len(pc_levels)
# Discrete colourscale: one flat band per category so each line takes its group colour
colorscale = []
for i in range(n):
    colorscale += [[i / n, palette[i % len(palette)]], [(i + 1) / n, palette[i % len(palette)]]]

par = go.Figure(go.Parcoords(
    line=dict(color=pc_codes, colorscale=colorscale, cmin=-0.5, cmax=n - 0.5),
    dimensions=dims,
))
par.update_layout(
    title=f"Parallel coordinates, coloured by {label(pc)} (drag along an axis to filter)",
    height=450, margin=dict(l=60, r=60, t=80, b=40),
)
st.plotly_chart(par, width="stretch")
st.caption("Each line is one record. Crossing lines between axes indicate negative correlation.")

# ── 2.6 Grand Tour (Animated 3D Scatter) ─────────────────────────────────────
st.header("2.6 Grand Tour (Animated 3D Scatter)")

xs, ys, zs = cfg["xyz"]
tour = px.scatter_3d(
    df, x=xs, y=ys, z=zs, color=cfg["color"],
    log_x=xs in cfg["log"], log_y=ys in cfg["log"], log_z=zs in cfg["log"],
    hover_data=cfg["tooltip_extra"], labels=LABELS, opacity=0.8,
    category_orders=ORDERS, color_discrete_sequence=palette,
)
tour.update_traces(marker=dict(size=4))

# A true grand tour projects through every 2D plane of the data; here we approximate it
# by orbiting the camera around the 3D cloud, which reveals structure from all sides.
angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
frames = [
    go.Frame(layout=dict(scene_camera=dict(eye=dict(x=1.8 * np.cos(t), y=1.8 * np.sin(t), z=0.8))))
    for t in angles
]
tour.frames = frames
tour.update_layout(
    title=f"{label(xs)} × {label(ys)} × {label(zs)} — press Play to rotate",
    height=600,
    scene_camera=dict(eye=dict(x=1.8, y=0, z=0.8)),
    updatemenus=[dict(
        type="buttons", showactive=False, x=0.05, y=1.0,
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True),
                                  transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
)
st.plotly_chart(tour, width="stretch")

# ── 3.3 New Visualization: Violin + Box Plot ─────────────────────────────────
st.header("3.3 New Visualization: Violin Plot with Box Overlay")

violin_y = st.selectbox("Variable to compare", numeric_cols, index=numeric_cols.index(cfg["heat_val"]))
violin = px.violin(
    df, x=cfg["cat_a"], y=violin_y, color=cfg["cat_a"], box=True, points="all",
    hover_data=cfg["tooltip_extra"], labels=LABELS, category_orders=ORDERS,
    color_discrete_sequence=palette, log_y=violin_y in cfg["log"],
    title=f"Distribution of {label(violin_y)} by {label(cfg['cat_a'])}",
)
violin.update_layout(height=480, showlegend=False)
st.plotly_chart(violin, width="stretch")
st.caption(
    "The violin shows the full distribution shape; the box inside marks the median and "
    "quartiles; dots are individual records."
)
