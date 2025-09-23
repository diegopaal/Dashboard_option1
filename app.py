# app.py
import re
import pandas as pd
import numpy as np
from textwrap import wrap
from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px

# =========================
# ---- Column aliases -----
# =========================
COL_L1 = "INTERVENTION_Level 1_One Health Capability"
COL_L2 = "INTERVENTION_Level 2"

COL_INT_OUT = "INTERMEDIATE OUTCOME_Classification (dropdown)\n"
COL_FIN_OUT = "OUTCOME_Classification (dropdown)\n\n"

COL_INT_STRENGTH = "_strength"
COL_FIN_STRENGTH = "_strength_final"
COL_INT_SIGN = "_sign"
COL_FIN_SIGN = "_sign_final"

# ---- Impact block ----
COL_IMP_OUT = "IMPACT_Classification (dropdown)\nAMR burden reduced = Impact"
COL_IMP_STRENGTH = "_strength_impact"
COL_IMP_SIGN = "_sign_impact"
COL_IMP_TEXT = "IMPACT TEXT (verbatim)"

COL_TITLE = "Title​ \n(213 articles)"
COL_YEAR = "Publication Year​"
COL_GEOG = "Geography​_Location"
COL_INT_TEXT = "INTERMEDIATE OUTCOME TEXT (verbatim)"
COL_INTERV_TEXT = "INTERVENTION TEXT (verbatim)\n"
COL_OUT_TEXT = "OUTCOME TEXT (verbatim)"

# =========================
# ---- LOAD YOUR DATA -----
# =========================
df = pd.read_parquet("df.parquet")

def _clean(s):
    return s.strip() if isinstance(s, str) else s

for c in [COL_L1, COL_L2, COL_INT_OUT, COL_FIN_OUT, COL_IMP_OUT]:
    if c in df.columns:
        df[c] = df[c].map(_clean)

# =========================
# ---- Long-format table ---
# =========================
keep_base = [
    COL_L1, COL_L2, COL_TITLE, COL_YEAR, COL_GEOG,
    COL_INT_TEXT, COL_INTERV_TEXT, COL_OUT_TEXT, COL_IMP_TEXT
]
cols_keep = [c for c in keep_base if c in df.columns]

# Intermediate
int_cols = cols_keep + [COL_INT_OUT, COL_INT_STRENGTH, COL_INT_SIGN]
int_long = (df[int_cols].dropna(subset=[COL_L1, COL_L2, COL_INT_OUT], how="any")
            .assign(outcome_type="Intermediate Outcomes",
                    outcome_name=lambda x: x[COL_INT_OUT],
                    strength=lambda x: x[COL_INT_STRENGTH],
                    sign=lambda x: x[COL_INT_SIGN]))

# Final
fin_cols = cols_keep + [COL_FIN_OUT, COL_FIN_STRENGTH, COL_FIN_SIGN]
fin_long = (df[fin_cols].dropna(subset=[COL_L1, COL_L2, COL_FIN_OUT], how="any")
            .assign(outcome_type="Final Outcomes",
                    outcome_name=lambda x: x[COL_FIN_OUT],
                    strength=lambda x: x[COL_FIN_STRENGTH],
                    sign=lambda x: x[COL_FIN_SIGN]))

# Impact (si faltaran columnas, crea bloque vacío)
if {COL_IMP_OUT, COL_IMP_STRENGTH, COL_IMP_SIGN}.issubset(df.columns):
    imp_cols = cols_keep + [COL_IMP_OUT, COL_IMP_STRENGTH, COL_IMP_SIGN]
    imp_long = (df[imp_cols].dropna(subset=[COL_L1, COL_L2, COL_IMP_OUT], how="any")
                .assign(outcome_type="Impact",
                        outcome_name=lambda x: x[COL_IMP_OUT],
                        strength=lambda x: x[COL_IMP_STRENGTH],
                        sign=lambda x: x[COL_IMP_SIGN]))
else:
    imp_long = pd.DataFrame(columns=int_long.columns)

long = pd.concat([int_long, fin_long, imp_long], ignore_index=True)

# =========================
# ---- Aggregation --------
# =========================
agg = (long.groupby([COL_L1, COL_L2, "outcome_type", "outcome_name"], dropna=False)
       .agg(n=("outcome_name", "size"),
            mean_strength=("strength", "mean"),
            mean_direction=("sign", "mean"))
       .reset_index())

# =========================
# ---- Utils --------------
# =========================
HDR_PREFIX_Y = "__HDR__"
ROW_PREFIX_Y = "__ROW__"
HDR_PREFIX_X = "__XHDR__"
COL_PREFIX_X = "__XCOL__"

INDENT = "\u2003\u2003"     # 2 EM spaces
BULLET = "•"

def wrap_lines(s, width=40, max_lines=3):
    s = "" if pd.isna(s) else str(s)
    lines = wrap(s, width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines-1] + [" ".join(lines[max_lines-1:])]
    return "\n".join(lines)

def make_row_label(s, width=40, max_lines=3):
    return f"{INDENT}{BULLET} {wrap_lines(s, width, max_lines)}"

def l2_num_prefix(s):
    m = re.match(r"\s*(\d+)[\.\)]", str(s))
    return int(m.group(1)) if m else 9999

# ----- Mapping de Strength (tus bandas) -----
def map_strength_label(x):
    if pd.isna(x):
        return "N/A"
    if x >= 2.75:
        return "High"
    elif 2.25 < x <= 2.75:
        return "Moderate-high"
    elif 1.75 < x <= 2.25:
        return "Moderate"
    elif 1.25 < x <= 1.75:
        return "Low-moderate"
    elif 0.75 < x <= 1.25:
        return "Low"
    elif 0 < x <= 1:
        return "Very low"

def map_direction_label(x):
    if pd.isna(x):
        return "N/A"
    if x == 1.0:
        return "Positive"
    elif 0.5 < x <= 1:
        return "Mostly positive"
    elif -0.5 < x <= 0.5:
        return "Inconclusive"
    elif -1 < x <= -0.5:
        return "Mostly negative"
    elif x == -1:
        return "Negative"

# Aplicar etiquetas al agregado (para hover)
agg["strength_label"] = agg["mean_strength"].map(map_strength_label)
agg["direction_label"] = agg["mean_direction"].map(map_direction_label)

# =========================
# ---- Hierarchical Y axis
# =========================
unique_l1 = list(agg[COL_L1].dropna().unique())
unique_l1 = [l for l in unique_l1 if str(l).lower().startswith("surveillance")] + \
            [l for l in unique_l1 if not str(l).lower().startswith("surveillance")]

agg["y_key"] = ROW_PREFIX_Y + agg[COL_L1].astype(str) + " || " + agg[COL_L2].astype(str)

order_y, y_ticktext, y_header_keys = [], [], []
for l1 in unique_l1:
    grp = agg[agg[COL_L1] == l1].copy()
    hdr_key = HDR_PREFIX_Y + str(l1)
    order_y.append(hdr_key)
    y_ticktext.append(" ")
    y_header_keys.append(hdr_key)
    grp = grp.sort_values(by=COL_L2, key=lambda s: s.map(l2_num_prefix))
    for _, r in grp.iterrows():
        order_y.append(r["y_key"])
        y_ticktext.append(make_row_label(r[COL_L2], width=40, max_lines=3))

order_y_rev  = order_y[::-1]
y_ticktext_rev = y_ticktext[::-1]

# =========================
# ---- Hierarchical X axis
# =========================
agg["x_key"] = COL_PREFIX_X + agg["outcome_type"] + " || " + agg["outcome_name"].astype(str)

order_x, x_ticktext, x_header_keys = [], [], []
x_items_by_type = {}

for t in ["Intermediate Outcomes", "Final Outcomes", "Impact"]:
    hdr = HDR_PREFIX_X + t
    order_x.append(hdr)
    x_ticktext.append("")
    x_header_keys.append(hdr)

    subnames = (agg.loc[agg["outcome_type"] == t, "outcome_name"]
                  .astype(str).sort_values().unique().tolist())
    x_items_by_type[t] = [COL_PREFIX_X + t + " || " + name for name in subnames]

    for key, name in zip(x_items_by_type[t], subnames):
        order_x.append(key)
        x_ticktext.append(make_row_label(name, width=24, max_lines=3))

# =========================
# ---- Labels para hover ---
# =========================
agg["Intervention_label"] = agg[COL_L1].astype(str) + " \u2192 " + agg[COL_L2].astype(str)
agg["Outcome_label"] = agg["outcome_name"].astype(str)

# =========================
# ---- Color bounds & sizing
# =========================
min_s = float(np.floor(np.nanmin(agg["mean_strength"].values) * 10) / 10) if len(agg) else 0.5
max_s = float(np.ceil(np.nanmax(agg["mean_strength"].values) * 10) / 10) if len(agg) else 3.0

n_rows = max(1, len(order_y))
fig_height = int(min(1400, 120 + 24 * n_rows))
fig_width = 1650

desired_max_px = 48
max_n = max(1, agg["n"].max())
sizeref = 2.0 * max_n / (desired_max_px ** 2)

# =========================
# ---- Figure -------------
# =========================
fig = px.scatter(
    agg, x="x_key", y="y_key", size="n", color="mean_strength",
    color_continuous_scale="RdYlGn", range_color=[min_s, max_s],
    size_max=desired_max_px,
    custom_data=[
        "Intervention_label",  # 0
        "Outcome_label",       # 1
        "n",                   # 2
        "strength_label",      # 3  ← texto
        "direction_label"      # 4  ← texto
    ]
)
fig.update_traces(
    marker=dict(line=dict(width=0), sizemode="area", sizeref=sizeref, sizemin=4),
    hovertemplate=(
        "<b>Intervention</b>: %{customdata[0]}<br>"
        "<b>Result</b>: %{customdata[1]}<br>"
        "<b>Number of articles</b>: %{customdata[2]}<br>"
        "<b>Mean Strength</b>: %{customdata[3]}<br>"
        "<b>Mean Direction</b>: %{customdata[4]}"
        "<extra></extra>"
    )
)

# Ticks del colorbar en los puntos medios de cada banda
colorbar_tickvals = [2.875, 2.5, 2.0, 1.5, 1.0, 0.5]
colorbar_ticktext = ["High", "Moderate-high", "Moderate", "Low-moderate", "Low", "Very low"]

fig.update_layout(
    height=fig_height,
    width=fig_width,
    margin=dict(l=520, r=60, t=90, b=330),
    xaxis=dict(
        title="",
        categoryorder="array", categoryarray=order_x,
        tickmode="array", tickvals=order_x, ticktext=x_ticktext,
        tickangle=35, tickfont=dict(size=11),
    ),
    yaxis=dict(
        title="Intervention themes",
        categoryorder="array", categoryarray=order_y_rev,
        tickmode="array", tickvals=order_y_rev, ticktext=y_ticktext_rev,
        automargin=True, tickfont=dict(size=11),
    ),
    coloraxis_colorbar=dict(
        title="Strength of evidence",
        tickmode="array",
        tickvals=colorbar_tickvals,
        ticktext=colorbar_ticktext
    ),
)

# ---- Row titles (Level 1)
for hdr_key in y_header_keys:
    l1_text = hdr_key.replace(HDR_PREFIX_Y, "")
    fig.add_annotation(
        xref="paper", yref="y",
        x=-0.0, y=hdr_key,
        text=f"<b>{wrap_lines(l1_text, 38, 3)}</b>",
        showarrow=False, xanchor="right", yanchor="middle",
        align="right", font=dict(size=13), bgcolor="rgba(0,0,0,0)"
    )

# ---- Column block titles (IO / FO / Impact)
N = len(order_x) - 1
for t in ["Intermediate Outcomes", "Final Outcomes", "Impact"]:
    items = x_items_by_type.get(t, [])
    if not items:
        continue
    first_idx = order_x.index(items[0])
    last_idx  = order_x.index(items[-1])
    center_frac = (first_idx + last_idx) / 2 / N

    fig.add_annotation(
        xref="x domain", yref="paper",
        x=center_frac, y=-0.25,
        text=f"<b>{t}</b>",
        textangle=0,
        showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=14)
    )

# =========================
# ---- Dash App ----------
# =========================
app = Dash(__name__)
app.title = "Scaling One Health Capabilities"

# Texto actualizado con tus bandas
app.layout = html.Div([
    html.H2("Scaling One Health Capabilities"),
    html.P(
        "Strength of Evidence is shown with a color gradient (red→green). "
        "Bands: Very low (0–1), Low (0.75–1.25), Low-moderate (1.25–1.75), "
        "Moderate (1.75–2.25), Moderate-high (2.25–2.75), High (≥2.75). "
        "Direction: Positive (+1), Inconclusive/Mixed (0), Negative (−1). "
        "Averages can be viewed by clicking on each bubble."
    ),
    dcc.Graph(id="bubble-graph", figure=fig, clear_on_unhover=True,
              style={"width": "100%"}),

    html.Hr(),
    html.H4(id="detail-title", children="Click a bubble to see the article list."),
    dash_table.DataTable(
        id="detail-table",
        columns=[
            {"name": "Title", "id": COL_TITLE},
            {"name": "Year", "id": COL_YEAR},
            {"name": "Location", "id": COL_GEOG},
            {"name": "Intervention", "id": COL_INTERV_TEXT},
            {"name": "Int. outcome", "id": COL_INT_TEXT},
            {"name": "Int. outcome - Strength", "id": COL_INT_STRENGTH},     # ← mostrará etiqueta
            {"name": "Int. outcome - Direction", "id": COL_INT_SIGN},
            {"name": "Final outcome", "id": COL_OUT_TEXT},
            {"name": "Final outcome - Strength", "id": COL_FIN_STRENGTH},     # ← mostrará etiqueta
            {"name": "Final outcome - Direction", "id": COL_FIN_SIGN},
            {"name": "Impact", "id": COL_IMP_TEXT},
            {"name": "Impact - Strength", "id": COL_IMP_STRENGTH},            # ← mostrará etiqueta
            {"name": "Impact - Direction", "id": COL_IMP_SIGN},
        ],
        data=[],
        sort_action="native", filter_action="native", page_size=10,
        style_cell={"whiteSpace": "pre-line", "textAlign": "left",
                    "minWidth": "160px", "maxWidth": "420px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_table={"overflowX": "auto"},
    )
], style={"maxWidth": "1800px", "margin": "0 auto", "padding": "12px"})

# =========================
# ---- Callbacks ----------
# =========================
@app.callback(
    Output("detail-table", "data"),
    Output("detail-title", "children"),
    Input("bubble-graph", "clickData"),
)
def show_details(clickData):
    if not clickData:
        return [], "Click a bubble to see the article list."

    x_key = clickData["points"][0]["x"]
    y_key = clickData["points"][0]["y"]

    if not (str(x_key).startswith(COL_PREFIX_X) and str(y_key).startswith(ROW_PREFIX_Y)):
        return [], "Select a bubble (row Level 2 × outcome) to see details."

    # Parse x: "__XCOL__<type> || <outcome>"
    _, rest = x_key.split(COL_PREFIX_X, 1)
    outcome_type, outcome_name = rest.split(" || ", 1)

    # Parse y: "__ROW__<L1> || <L2>"
    _, rest_y = y_key.split(ROW_PREFIX_Y, 1)
    l1, l2 = rest_y.split(" || ", 1)

    mask = ((long[COL_L1] == l1) &
            (long[COL_L2] == l2) &
            (long["outcome_type"] == outcome_type) &
            (long["outcome_name"] == outcome_name))
    sub = long.loc[mask].copy()

    # Asegurar columnas esperadas
    cols_show = [
        COL_TITLE, COL_YEAR, COL_GEOG, COL_INTERV_TEXT,
        COL_INT_TEXT, COL_OUT_TEXT, COL_IMP_TEXT,
        COL_INT_STRENGTH, COL_FIN_STRENGTH, COL_IMP_STRENGTH,
        COL_INT_SIGN, COL_FIN_SIGN, COL_IMP_SIGN
    ]
    for c in cols_show:
        if c not in sub.columns:
            sub[c] = np.nan

    # ---- Mapea los Strength numéricos a etiquetas para la tabla ----
    if COL_INT_STRENGTH in sub.columns:
        sub[COL_INT_STRENGTH] = sub[COL_INT_STRENGTH].map(map_strength_label)
    if COL_FIN_STRENGTH in sub.columns:
        sub[COL_FIN_STRENGTH] = sub[COL_FIN_STRENGTH].map(map_strength_label)
    if COL_IMP_STRENGTH in sub.columns:
        sub[COL_IMP_STRENGTH] = sub[COL_IMP_STRENGTH].map(map_strength_label)

    # (Opcional) También puedes mapear Direction a texto si quieres consistencia visual:
    # sub[COL_INT_SIGN] = sub[COL_INT_SIGN].map(map_direction_label)
    # sub[COL_FIN_SIGN] = sub[COL_FIN_SIGN].map(map_direction_label)
    # sub[COL_IMP_SIGN] = sub[COL_IMP_SIGN].map(map_direction_label)

    title = f'{outcome_type} — "{outcome_name}"  |  {l1} → {l2}  |  {len(sub)} articles'
    return sub[cols_show].to_dict("records"), title

# =========================
# ---- Run server ---------
# =========================
if __name__ == "__main__":
    app.run(debug=True)


