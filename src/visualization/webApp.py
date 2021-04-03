"""Streamlit web app to display Icestupa class object
"""

# External modules
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
import logging
import coloredlogs

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa
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
        "Select Icestupa",
        ("Schwarzsee 2019", "Guttannen 2020", "Guttannen 2021"),
    )
    trigger = st.sidebar.radio(
        "Select Fountain control", ("None", "Manual", "Temperature", "Weather")
    )
    SITE, FOUNTAIN, FOLDER = config(location, trigger=trigger)

    if (
        SITE["name"] in ["guttannen20"]
        and FOUNTAIN["trigger"] == "Manual"
    ):
        st.error(
            "Sorry, manual fountain control not recorded. Please choose a different fountain control"
        )
    else:
        icestupa = Icestupa(SITE, FOUNTAIN, FOLDER)
        icestupa.read_output()
        df_in = icestupa.df
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
        df_in = df_in.set_index("When")
        df = df_in

        input_folder = os.path.join(dirname, "data/" + SITE["name"] + "/interim/")
        output_folder = os.path.join(dirname, "data/" + SITE["name"] + "/processed/")
        col1, mid, col2 = st.beta_columns([4, 6, 20])
        (
            input_cols,
            input_vars,
            output_cols,
            output_vars,
            derived_cols,
            derived_vars,
        ) = vars(df_in)

        with col1:
            air_logo = os.path.join(dirname, "src/visualization/AIR_logo.png")
            st.image(air_logo, width=180)
        with col2:
            st.write("## Artificial Ice Reservoir Simulation")
            if trigger == "None":
                st.write(
                    "### Fountain was always kept on until **%s** "
                    % (icestupa.fountain_off_date.date())
                )
            if trigger == "Manual":
                st.write(
                    "### Fountain was controlled **%s** until **%s**"
                    % (trigger + "ly", (icestupa.fountain_off_date.date()))
                )
            if trigger == "Temperature":
                st.write(
                    "### Fountain was switched on/off after sunset when temperature was below **%s** until **%s**"
                    % (icestupa.crit_temp, (icestupa.fountain_off_date.date()))
                )
            if trigger == "Weather":
                st.write(
                    "### Fountain was switched on/off whenever surface energy balance was negative/positive respectively until **%s**"
                    % (icestupa.fountain_off_date.date())
                )

        # df = df_filter("Move sliders to filter dataframe", icestupa.df)

        st.sidebar.write("Display Variables")
        timelapse = st.sidebar.checkbox("Timelapse", value=True)
        summary = st.sidebar.checkbox("Summary", value=True)
        input = st.sidebar.checkbox("Input")
        output = st.sidebar.checkbox("Output")
        derived = st.sidebar.checkbox("Derived")
        st.sidebar.write("# Map of %s" % location)
        lat = SITE["latitude"]
        lon = SITE["longitude"]
        map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.sidebar.map(map_data, zoom=10)
        if timelapse:
            if location == "Schwarzsee 2019":
                st.write("## %s Timelapse" % (location))
                url = "https://youtu.be/GhljRBGpxMg"
            if location == "Guttannen 2021":
                st.write("## %s Timelapse" % (location))
                url = "https://youtu.be/DBHoL1Z7H6U"
            if location == "Guttannen 2020":
                st.write("## %s Timelapse" % (location))
                url = "https://youtu.be/kcrvhU20OOE"

            st.video(url)
            st.write("## Volume Estimation and Validation")
            fig, ax = plt.subplots()
            CB91_Blue = "#2CBDFE"
            CB91_Green = "#47DBCD"
            x = icestupa.df.When
            y1 = icestupa.df.iceV
            y2 = icestupa.df.DroneV
            ax.set_ylabel("Ice Volume[$m^3$]")
            ax.plot(
                x,
                y1,
                "b-",
                label="Modelled Volume",
                linewidth=1,
                color=CB91_Blue,
            )
            ax.scatter(x, y2, color=CB91_Green, label="Measured Volume")
            ax.set_ylim(bottom=0)
            plt.legend()
            ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            fig.autofmt_xdate()
            st.pyplot(fig)

            if SITE["name"] in ["guttannen21", "guttannen20"] :
                fig, ax = plt.subplots()
                CB91_Purple = "#9D2EC5"
                CB91_Violet = "#661D98"
                CB91_Amber = "#F5B14C"
                x = icestupa.df.When
                y1 = icestupa.df.T_s
                y2 = icestupa.df.cam_temp
                ax.plot(
                    x,
                    y1,
                    "b-",
                    label="Modelled Temperature",
                    linewidth=1,
                    color=CB91_Amber,
                    zorder=0,
                )
                ax.scatter(x, y2, color=CB91_Violet, s=1, label="Measured Temperature", zorder=1)#, marker='+')
                # ax.set_ylim(bottom=0)
                plt.legend()
                ax.xaxis.set_major_locator(mdates.WeekdayLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                ax.xaxis.set_minor_locator(mdates.DayLocator())
                fig.autofmt_xdate()
                st.pyplot(fig)

        if summary:
            st.write("### Maximum Ice Volume: %.2f m3" % icestupa.df["iceV"].max())
            st.write(
                "### Meltwater Released: %.2f litres"
                % icestupa.df["meltwater"].iloc[-1]
            )
            st.write("## Input variables")
            st.image(output_folder + "paper_figures/Model_Input_" + trigger + ".jpg")
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
                variable_out = [
                    output_vars[output_cols.index(item)] for item in variable2
                ]
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
                variable_in = [
                    derived_vars[derived_cols.index(item)] for item in variable3
                ]
                variable = variable_in
                for v in variable:

                    meta = get_parameter_metadata(v)
                    # st.header("%s %s" % (meta["kind"], meta["name"] + " " + meta["units"]))
                    st.header("%s" % (meta["name"] + " " + meta["units"]))
                    st.line_chart(df[v])
