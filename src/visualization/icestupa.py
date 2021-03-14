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

# import matplotlib.dates as mdates
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.offsetbox import AnchoredText
# from matplotlib.ticker import AutoMinorLocator
# from matplotlib.backends.backend_pdf import PdfPages

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)
import re
import base64

import logging
import coloredlogs
from src.models.air import Icestupa, PDF
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


@st.cache
def vars(df_in):
    input_cols = []
    input_vars = []
    output_cols = []
    output_vars = []
    for variable in df_in.columns:
        v = icestupa.get_parameter_metadata(variable)
        if v["kind"] == "Input":
            input_cols.append(v["name"])
            input_vars.append(variable)
        if v["kind"] == "Output":
            output_cols.append(v["name"])
            output_vars.append(variable)
    return input_cols, input_vars, output_cols, output_vars


if __name__ == "__main__":
    location = st.sidebar.radio(
        # "Select Location", ("Gangles", "Schwarzsee", "Guttannen", "Hial", "Secmol")
        "Select Location",
        ("Schwarzsee", "Gangles", "Guttannen"),
    )
    SITE, FOUNTAIN = config(location)
    lat = SITE["latitude"]
    lon = SITE["longitude"]
    map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
    start_date = SITE["start_date"]
    h_f = FOUNTAIN["h_f"]

    if location == "Schwarzsee":
        trigger = st.sidebar.radio(
            "Select Discharge Trigger", ("Manual", "Temperature", "NetEnergy")
        )
        FOUNTAIN["trigger"] = trigger
        icestupa = PDF(SITE, FOUNTAIN)
        icestupa.read_output()
        df_in = icestupa.df
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
        df_in = df_in.set_index("When")
        df = df_in

    if location == "Gangles":
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
        df_in = icestupa.df
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
        df_in = df_in.set_index("When")
        df = df_in
        # icestupa.summary()

    if location == "Guttannen":
        trigger = st.sidebar.radio(
            "Select Discharge Trigger", ("Manual", "Temperature", "NetEnergy")
        )
        FOUNTAIN["trigger"] = trigger
        icestupa = PDF(SITE, FOUNTAIN)

        input_folder = os.path.join(dirname, "data/" + "guttannen" + "/interim/")
        input_file = input_folder + "guttannen" + "_input_model.csv"
        df = pd.read_csv(input_file, sep=",", header=0, parse_dates=["When"])
        df_in = df.set_index("When")
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
        df = df_in

    st.header(
        "**%s** site with fountain discharge triggered by **%s**" % (location, trigger)
    )

    input_cols, input_vars, output_cols, output_vars = vars(df_in)

    # df = df_filter("Move sliders to filter dataframe", icestupa.df)

    variable1 = st.sidebar.multiselect(
        "Choose Input variables",
        options=(input_cols),
        # default=["Fountain Spray", "Temperature"],
        default=["Temperature"],
    )
    variable2 = st.sidebar.multiselect(
        "Choose Output variables",
        options=(output_cols),
        # default=["Ice Volume"],
    )
    st.sidebar.map(map_data, zoom=10)

    if not (variable1 or variable2):
        st.error("Please select at least one variable.")
    else:
        variable_in = [input_vars[input_cols.index(item)] for item in variable1]
        variable_out = [output_vars[output_cols.index(item)] for item in variable2]
        variable = variable_in + variable_out
        for v in variable:
            meta = icestupa.get_parameter_metadata(v)
            st.header("%s %s" % (meta["kind"], meta["name"] + " " + meta["units"]))
            st.line_chart(df[v])

            st.markdown(download_csv(meta["name"], df[v]), unsafe_allow_html=True)
