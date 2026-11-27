"""
Page themes for the Streamlit app: three backgrounds, matching text colors,
and one colorful chart palette. Used by app.py, lab.py and other_tools.py.
"""
import streamlit as st

THEMES = {
    "Light": {"bg": "#E7E7E7", "text": "#111111", "input": "#F4F4F4", "template": "plotly_white"},
    "Mid grey": {"bg": "#868686", "text": "#111111", "input": "#9A9A9A", "template": "plotly_white"},
    "Dark": {"bg": "#0D0D0D", "text": "#EDEDED", "input": "#1C1C1C", "template": "plotly_dark"},
}
DEFAULT_THEME = "Light"

# Chart colors, used in this order
PALETTE = ["#2563EB", "#E4572E", "#17A398", "#F2A541", "#7B2CBF", "#D7263D"]

FONT = "Georgia, 'Times New Roman', serif"  # research-paper look


# Turn "#RRGGBB" into "rgba(r, g, b, alpha)" so we can make faint versions of a color.
def faded(hex_color, alpha):
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


# Solid color that is `share` of color1 and the rest color2 (e.g. 15% text on the background).
def mix(color1, color2, share):
    channels = []
    for start in (1, 3, 5):
        value1 = int(color1[start:start + 2], 16)
        value2 = int(color2[start:start + 2], 16)
        channels.append(round(share * value1 + (1 - share) * value2))
    return "#{:02X}{:02X}{:02X}".format(*channels)


# Colors for the theme saved in the session (Light if none chosen yet).
def current_colors():
    name = st.session_state.get("theme", DEFAULT_THEME)
    colors = dict(THEMES[name])
    colors["name"] = name
    colors["grid"] = mix(colors["text"], colors["bg"], 0.15)    # gridlines: very faint text color
    colors["border"] = mix(colors["text"], colors["bg"], 0.30)  # borders: a bit stronger
    return colors


# Sidebar radio buttons for the theme. The choice is copied to
# st.session_state["theme"] so it survives switching pages.
# A "?theme=Dark" (or Mid grey / Light) link sets the starting theme.
def theme_picker():
    if "theme" not in st.session_state:
        from_link = st.query_params.get("theme", DEFAULT_THEME)
        st.session_state["theme"] = from_link if from_link in THEMES else DEFAULT_THEME
    names = list(THEMES)
    choice = st.sidebar.radio("Theme", names, index=names.index(st.session_state["theme"]),
                              horizontal=True)
    st.session_state["theme"] = choice


# Inject CSS so the whole page (not just the charts) uses the theme colors.
def apply_theme():
    c = current_colors()
    st.markdown(f"""
<style>
.stApp, [data-testid="stHeader"] {{ background-color: {c['bg']}; }}
[data-testid="stSidebar"] {{ background-color: {c['bg']}; border-right: 1px solid {c['border']}; }}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp td, .stApp th, .stApp caption, [data-testid="stMetricValue"] {{
    color: {c['text']}; font-family: {FONT};
}}
.stApp h1, .stApp h2, .stApp h3 {{ font-weight: 500; letter-spacing: 0.01em; }}
[data-testid="stCaptionContainer"], .stApp small {{ color: {c['text']}; opacity: 0.75; }}
/* Inputs */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="input"] > div,
.stApp input, .stApp textarea, [data-testid="stNumberInputContainer"],
[data-testid="stFileUploaderDropzone"] {{
    background-color: {c['input']} !important; color: {c['text']} !important;
    border-color: {c['border']} !important;
}}
/* Pills (expiration picker): unselected = outlined, selected = palette blue */
[data-testid="stButtonGroup"] button {{ background-color: transparent; color: {c['text']};
    border: 1px solid {c['border']}; }}
[data-testid="stButtonGroup"] button[kind="pillsActive"] {{ background-color: {PALETTE[0]};
    border-color: {PALETTE[0]}; }}
[data-testid="stButtonGroup"] button[kind="pillsActive"] p {{ color: #FFFFFF; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li {{
    background-color: {c['input']} !important; color: {c['text']} !important;
}}
.stDownloadButton button, .stButton button, [data-testid="stNumberInputContainer"] button {{
    background-color: {c['input']}; color: {c['text']}; border: 1px solid {c['border']}; }}
/* Tables (st.table) */
.stApp table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
.stApp th, .stApp td {{ border: none !important; border-bottom: 1px solid {c['border']} !important;
    background-color: transparent !important; padding: 0.35rem 0.75rem !important; }}
.stApp th {{ font-weight: 600; }}
.stApp tbody th, .stApp tbody th p {{ white-space: nowrap !important; }}  /* keep dates on one line */
/* Thin rules and boxes */
hr {{ border-color: {c['border']} !important; }}
[data-testid="stExpander"] details {{ border: 1px solid {c['border']}; border-radius: 4px; }}
[data-testid="stAlert"] {{ border: 1px solid {c['border']}; }}
[data-testid="stAlert"] * {{ color: {c['text']} !important; }}
.block-container {{ padding-top: 2.5rem; max-width: 1200px; }}
</style>
""", unsafe_allow_html=True)


# Apply theme colors, fonts and thin gridlines to a Plotly chart.
def style_chart(fig, height=450):
    c = current_colors()
    fig.update_layout(
        template=c["template"], paper_bgcolor=c["bg"], plot_bgcolor=c["bg"],
        font=dict(color=c["text"], family=FONT, size=13), height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(y=0.97, yanchor="top", font=dict(size=15)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=c["input"], font_color=c["text"]),
    )
    fig.update_xaxes(gridcolor=c["grid"], linecolor=c["border"], zeroline=False, showline=True)
    fig.update_yaxes(gridcolor=c["grid"], linecolor=c["border"], zeroline=False, showline=True)
    return fig
