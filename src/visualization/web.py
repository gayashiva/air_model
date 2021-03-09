import streamlit as st
import altair as alt
import pandas as pd
import sys
from datetime import datetime
from tqdm import tqdm
import os

os.environ[
    "TZ"
] = "UTC"  # Hardcoded as workaround to  https://github.com/streamlit/streamlit/issues/106
import numpy as np
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)
import re
import base64

import logging
import coloredlogs
from src.models.air import Icestupa
from src.data.settings import config


def download_csv(name, df):

    csv = df.to_csv()
    base = base64.b64encode(csv.encode()).decode()
    file = (
        f'<a href="data:file/csv;base64,{base}" download="%s.csv">Download file</a>'
        % (name)
    )

    return file


def df_filter(message, df):

    slider_1, slider_2 = st.sidebar.slider(
        "%s" % (message), 0, len(df) - 1, [0, len(df) - 1], 1
    )

    while len(str(df.iloc[slider_1][0]).replace(".0", "")) < 4:
        df.iloc[slider_1, 1] = "0" + str(df.iloc[slider_1][1]).replace(".0", "")

    while len(str(df.iloc[slider_2][0]).replace(".0", "")) < 4:
        df.iloc[slider_2, 1] = "0" + str(df.iloc[slider_1][1]).replace(".0", "")

    start_date = datetime.strptime(
        str(df.iloc[slider_1][0]).replace(".0", ""),
        "%Y-%m-%d %H:%M:%S",
    )
    start_date = start_date.strftime("%d %b %Y, %I:%M%p")

    end_date = datetime.strptime(
        str(df.iloc[slider_2][0]).replace(".0", ""),
        "%Y-%m-%d %H:%M:%S",
    )
    end_date = end_date.strftime("%d %b %Y, %I:%M%p")

    st.info("Start: **%s** End: **%s**" % (start_date, end_date))

    filtered_df = df.iloc[slider_1 : slider_2 + 1][:].reset_index(drop=True)

    return filtered_df


if __name__ == "__main__":
    location = st.sidebar.radio(
        # "Select Location", ("Gangles", "Schwarzsee", "Guttannen", "Hial", "Secmol")
        "Select Location",
        ("Schwarzsee", "Gangles"),
    )
    SITE, FOUNTAIN = config(location)
    start_date = SITE["start_date"]
    h_f = FOUNTAIN["h_f"]
    if location != "Schwarzsee":
        trigger = st.sidebar.radio(
            "Select Discharge Trigger", ("Temperature", "NetEnergy")
        )
        FOUNTAIN["trigger"] = trigger
        start_date = st.date_input("Fountain spray starts at", start_date)
        start_date = pd.to_datetime(start_date)
        h_f = st.number_input("Fountain height starts at", value=h_f, min_value=1)
        if start_date > SITE["start_date"] or h_f != FOUNTAIN["h_f"]:

            SITE["start_date"] = pd.to_datetime(start_date)
            FOUNTAIN["h_f"] = h_f

            icestupa = Icestupa(SITE, FOUNTAIN)

            icestupa.derive_parameters()

            icestupa.melt_freeze()
            # icestupa.summary()
        else:
            icestupa = Icestupa(SITE, FOUNTAIN)
            icestupa.read_output()
    else:
        trigger = st.sidebar.radio(
            "Select Discharge Trigger", ("Temperature", "NetEnergy", "Manual")
        )
        FOUNTAIN["trigger"] = trigger
        icestupa = Icestupa(SITE, FOUNTAIN)
        icestupa.read_output()

    # mode = st.sidebar.radio("Mode", ("Normal", "Advanced"))

    st.header(
        "**%s** site with fountain discharge triggered by **%s**" % (location, trigger)
    )

    # if mode == "Advanced":
    # else:
    #     icestupa = Icestupa(SITE, FOUNTAIN)
    #     icestupa.read_output()

    df_in = icestupa.df[
        [
            "When",
            # "sea",
            "T_a",
            "RH",
            "v_a",
            "Discharge",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "p_a",
            # "cld",
            "a",
            "vp_a",
            "LW_in",
            "T_s",
            "T_bulk",
            "f_cone",
            "ice",
            "iceV",
            # "solid",
            # "gas",
            # "vapour",
            # "melted",
            # "delta_T_s",
            "unfrozen_water",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "meltwater",
            "SA",
            "h_ice",
            "r_ice",
            "ppt",
            "dpt",
            "cdt",
            # "missing",
            "s_cone",
            # "input",
            "vp_ice",
            "thickness",
            "$q_{T}$",
            "$q_{melt}$",
        ]
    ]
    df_in = df_in.set_index("When")
    cols = [
        icestupa.get_parameter_metadata(item)["name"] for item in df_in.columns.tolist()
    ]
    df = df_in

    # df = df_filter("Move sliders to filter dataframe", icestupa.df)

    variable = st.sidebar.multiselect(
        "Choose Input/Output variables",
        (cols),
        ["Ice Volume", "Fountain Spray", "Temperature"],
    )

    if not variable:
        st.error("Please select at least one variable.")
    else:
        variable = [df.columns[cols.index(item)] for item in variable]
        for v in variable:
            meta = icestupa.get_parameter_metadata(v)
            st.header("%s %s" % (meta["kind"], meta["name"] + " " + meta["units"]))
            st.line_chart(df[v])
            st.markdown(download_csv(meta["name"], df[v]), unsafe_allow_html=True)

# df["When"] = df["When"].dt.tz_localize(
#     "UTC"
# )  # Hardcoded as workaround to  https://github.com/streamlit/streamlit/issues/1061
