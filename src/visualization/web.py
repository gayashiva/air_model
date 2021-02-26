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


# @st.cache(allow_output_mutation=True)
def load_data(trigger="Schwarzsee"):
    SITE, FOUNTAIN = config(location)
    FOUNTAIN["trigger"] = trigger
    icestupa = Icestupa(SITE, FOUNTAIN)
    icestupa.read_output()
    # st.write(icestupa.df.Discharge.head())
    # st.write(icestupa.trigger)
    return icestupa.df, SITE, FOUNTAIN


def download_csv(name, df):

    csv = df.to_csv(index=False)
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


def config(location="Schwarzsee"):

    if location == "Schwarzsee":
        SITE = dict(
            name="schwarzsee",
            end_date=datetime(2019, 3, 17),
            start_date=datetime(2019, 1, 30, 17),
            utc_offset=1,
            longitude=7.297543,
            latitude=46.693723,
            h_aws=3,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2019, 3, 10, 18),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=1.35,  # FOUNTAIN steps h_f
            theta_f=45,  # FOUNTAIN aperture diameter
            ftl=0,  # FOUNTAIN flight time loss ftl
            T_w=5,  # FOUNTAIN Water temperature
            discharge=3.58,  # FOUNTAIN on discharge
            crit_temp=0,  # FOUNTAIN runtime temperature
        )

    if location == "Leh":
        SITE = dict(
            name="leh",
            end_date=datetime(2019, 4, 9),
            start_date=datetime(2019, 1, 30, 17),
            utc_offset=1,
            longitude=77.5771,
            latitude=34.1526,
            h_aws=3,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2019, 2, 16, 10),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=1.35,  # FOUNTAIN steps h_f
            theta_f=45,  # FOUNTAIN aperture diameter
            ftl=0,  # FOUNTAIN flight time loss ftl
            T_w=5,  # FOUNTAIN Water temperature
            discharge=3.58,  # FOUNTAIN on discharge
            crit_temp=0,  # FOUNTAIN runtime temperature
        )

    if location == "Schwarzsee_2020":
        dates = dict(
            start_date=datetime(2020, 2, 15),
            end_date=datetime(2020, 2, 18),
            fountain_off_date=datetime(2020, 2, 10),
        )
        FOUNTAIN = dict(
            aperture_f=0.005,  # FOUNTAIN aperture diameter
            h_f=4,  # FOUNTAIN steps h_f
            discharge=3.58,  # FOUNTAIN on discharge
            crit_temp=-5,  # FOUNTAIN runtime temperature
            latitude=46.693723,
            longitude=7.297543,
            utc_offset=1,
        )

    if location == "Guttannen":

        SITE = dict(
            name="guttannen",
            start_date=datetime(2020, 1, 1, 18),
            end_date=datetime(2020, 5, 1),
            error_date=datetime(2020, 1, 19),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=3,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2019, 3, 1),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=3.93,  # FOUNTAIN steps h_f
            theta_f=0,  # FOUNTAIN aperture diameter
            ftl=0,  # FOUNTAIN flight time loss ftl
            T_w=5,  # FOUNTAIN Water temperature
            discharge=3.58,  # FOUNTAIN on discharge
            crit_temp=0,  # FOUNTAIN runtime temperature
            tree_height=1.93,
            tree_radius=4.13 / 2,
        )

    # FOLDERS = dict(
    #     raw_folder=os.path.join(dirname, "data/" + SITE["name"] + "/raw/"),
    #     input_folder=os.path.join(dirname, "data/" + SITE["name"] + "/interim/"),
    #     output_folder=os.path.join(dirname, "data/" + SITE["name"] + "/processed/"),
    #     sim_folder=os.path.join(
    #         dirname, "data/" + SITE["name"] + "/processed/simulations"
    #     ),
    # )
    return SITE, FOUNTAIN


input_full_names = dict(
    Discharge="Fountain Spray ($l\\, min^{-1}$)",
    T_a="Temperature ($\\degree C$)",
    RH="Humidity ($\\%$)",
    p_a="Pressure ($hPa$)",
    SW_direct="Shortwave Direct ($W\\,m^{-2}$)",
    SW_diffuse="Shortwave Diffuse ($W\\,m^{-2}$)",
    LW_in="Longwave Radiation ($W\\,m^{-2}$)",
    Prec="Precipitation ($mm$)",
    v_a="Wind speed ($m\\,s^{-1}$)",
)
output_full_names = dict(
    iceV="Ice Volume",
    ice="Ice Mass",
    a="Albedo",
    thickness="Ice thickness",
)

if __name__ == "__main__":
    location = st.sidebar.radio("Select Location", ("Schwarzsee", "Guttannen"))
    fountain = st.sidebar.radio(
        "Select Discharge Trigger", ("Schwarzsee", "Temperature", "NetEnergy")
    )

    st.header("**%s** with fountain discharge trigger **%s**" % (location, fountain))
    df_i, SITE, FOUNTAIN = load_data(fountain)

    df = df_filter("Move sliders to filter dataframe", df_i)

    # column_1, column_2 = st.beta_columns(2)
    # with column_1:
    #     st.header("Location")
    #     st.write(SITE)

    # with column_2:
    #     st.header("Fountain")
    #     st.write(FOUNTAIN)

    df = df.set_index("When")
    variable = st.sidebar.multiselect(
        "Choose input/output variables", (df.columns.tolist()), ["Discharge", "iceV"]
    )
    if not variable:
        st.error("Please select at least one variable.")
    else:
        data = df[variable]
        for v in variable:
            if v in input_full_names.keys():
                st.header("Input %s" % (input_full_names[v]))
                # st.markdown(download_csv(full_names[v], df[v]), unsafe_allow_html=True)
                st.line_chart(df[v])
            if v in output_full_names.keys():
                st.header("Output %s" % (output_full_names[v]))
                # st.markdown(download_csv(full_names[v], df[v]), unsafe_allow_html=True)
                st.line_chart(df[v])

# df["When"] = df["When"].dt.tz_localize(
#     "UTC"
# )  # Hardcoded as workaround to  https://github.com/streamlit/streamlit/issues/1061
