import streamlit as st
import pandas as pd
import sys
from datetime import datetime
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import re
import base64

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)

from src.models.methods.metadata import get_parameter_metadata
from src.models.air import Icestupa
from src.data.settings import config

import logging
import coloredlogs


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
    derived_cols = []
    derived_vars = []
    for variable in df_in.columns:
        v = get_parameter_metadata(variable)
        if v["kind"] == "Input":
            input_cols.append(v["name"])
            input_vars.append(variable)
        if v["kind"] == "Output":
            output_cols.append(v["name"])
            output_vars.append(variable)
        if v["kind"] == "Derived":
            derived_cols.append(v["name"])
            derived_vars.append(variable)
    return input_cols, input_vars, output_cols, output_vars, derived_cols, derived_vars


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    location = st.sidebar.radio(
        # "Select Location", ("Gangles", "Schwarzsee", "Guttannen", "Hial", "Secmol")
        "Select Location",
        ("Guttannen", "Schwarzsee"),
    )
    SITE, FOUNTAIN, FOLDER = config(location)
    lat = SITE["latitude"]
    lon = SITE["longitude"]
    map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
    start_date = SITE["start_date"]
    h_f = FOUNTAIN["h_f"]

    if location in ["Guttannen", "Schwarzsee"]:
        trigger = st.sidebar.radio("Select Discharge Trigger", ("Manual", "NetEnergy"))
        FOUNTAIN["trigger"] = trigger
        icestupa = Icestupa(SITE, FOUNTAIN, FOLDER)
        icestupa.read_output()
        df_in = icestupa.df
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
        df_in = df_in.set_index("When")
        df = df_in
    # if location == "Schwarzsee":
    #     trigger = st.sidebar.radio(
    #         "Select Discharge Trigger", ("Manual", "NetEnergy")
    #     )
    #     FOUNTAIN["trigger"] = trigger
    #     icestupa = Icestupa(SITE, FOUNTAIN)
    #     icestupa.read_output()
    #     df_in = icestupa.df
    #     df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
    #     df_in = df_in.set_index("When")
    #     df = df_in

    if location == "Gangles":
        trigger = st.sidebar.radio("Select Discharge Trigger", ("NetEnergy"))
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

    input_folder = os.path.join(dirname, "data/" + SITE["name"] + "/interim/")
    output_folder = os.path.join(dirname, "data/" + SITE["name"] + "/processed/")
    col1, mid, col2 = st.beta_columns([4, 6, 20])
    input_cols, input_vars, output_cols, output_vars, derived_cols, derived_vars = vars(
        df_in
    )

    with col1:
        air_logo = os.path.join(dirname, "src/visualization/AIR_logo.png")
        st.image(air_logo, width=180)
    with col2:
        st.write("## Artificial Ice Reservoir Simulation")
        st.write("## **%s** Icestupa " % (location))
        st.write("### **%s** Fountain trigger" % (trigger))

    # df = df_filter("Move sliders to filter dataframe", icestupa.df)

    st.sidebar.write("Display Variables")
    timelapse = st.sidebar.checkbox("Timelapse", value=True)
    summary = st.sidebar.checkbox("Summary", value=True)
    input = st.sidebar.checkbox("Input")
    output = st.sidebar.checkbox("Output")
    derived = st.sidebar.checkbox("Derived")
    st.sidebar.write("# Map of %s" % location)
    st.sidebar.map(map_data, zoom=10)
    if timelapse:
        if location == "Schwarzsee":
            st.write("## %s Timelapse" % (location))
            url = "https://youtu.be/GhljRBGpxMg"
        if location == "Guttannen":
            st.write("## %s Timelapse" % (location))
            url = "https://youtu.be/DBHoL1Z7H6U"
        st.video(url)
        st.write("## Volume Estimation and Validation")
        fig, ax = plt.subplots()
        ax.set_ylabel("Ice Volume[$m^3$]")
        CB91_Blue = "#2CBDFE"
        CB91_Green = "#47DBCD"
        x = icestupa.df.When
        y1 = icestupa.df.iceV
        y2 = icestupa.df.DroneV
        ax.plot(
            x,
            y1,
            "b-",
            label="Modelled Ice Volume",
            linewidth=1,
            color=CB91_Blue,
        )
        ax.scatter(x, y2, color=CB91_Green, label="Drone Volume")
        ax.set_ylim(bottom=0)
        plt.legend()
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        fig.autofmt_xdate()
        st.pyplot(fig)

    if summary:
        st.write("### Maximum Ice Volume: %.2f m3" % icestupa.df["iceV"].max())
        st.write(
            "### Meltwater Released: %.2f litres" % icestupa.df["meltwater"].iloc[-1]
        )
        Duration = icestupa.df.index[-1] * icestupa.TIME_STEP / (60 * 60 * 24)
        st.write("### Survival Duration:  %.2f days" % round(Duration, 2))
        col1, mid, col2 = st.beta_columns([14, 2, 14])
        with col1:
            st.write("## Input variables")
            st.image(output_folder + "paper_figures/Model_Input_" + trigger + ".jpg")

        with col2:
            st.write("## Output variables")
            st.image(output_folder + "paper_figures/Model_Output_" + trigger + ".jpg")

    if input:
        st.write("## Input variables")
        variable1 = st.multiselect(
            "Choose",
            options=(input_cols),
            # default=["Fountain Spray", "Temperature"],
            default=["Temperature"],
        )
        if not (variable1):
            st.error("Please select at least one variable.")
        else:
            variable_in = [input_vars[input_cols.index(item)] for item in variable1]
            variable = variable_in
            for v in variable:

                meta = get_parameter_metadata(v)
                # st.header("%s %s" % (meta["kind"], meta["name"] + " " + meta["units"]))
                st.header("%s" % (meta["name"] + " " + meta["units"]))
                st.line_chart(df[v])

                # st.markdown(download_csv(meta["name"], df[v]), unsafe_allow_html=True)

    if output:
        st.write("## Output variables")

        variable2 = st.multiselect(
            "Choose",
            options=(output_cols),
            default=["Ice Volume"],
        )
        if not (variable2):
            st.error("Please select at least one variable.")
        else:
            variable_out = [output_vars[output_cols.index(item)] for item in variable2]
            variable = variable_out
            for v in variable:
                meta = get_parameter_metadata(v)
                st.header("%s" % (meta["name"] + " " + meta["units"]))
                st.line_chart(df[v])
                # st.markdown(download_csv(meta["name"], df[v]), unsafe_allow_html=True)
    if derived:
        st.write("## Derived variables")
        variable3 = st.multiselect(
            "Choose",
            options=(derived_cols),
            # default=["Fountain Spray", "Temperature"],
            # default=["Temperature"],
        )
        if not (variable3):
            st.error("Please select at least one variable.")
        else:
            variable_in = [derived_vars[derived_cols.index(item)] for item in variable3]
            variable = variable_in
            for v in variable:

                meta = get_parameter_metadata(v)
                # st.header("%s %s" % (meta["kind"], meta["name"] + " " + meta["units"]))
                st.header("%s" % (meta["name"] + " " + meta["units"]))
                st.line_chart(df[v])
