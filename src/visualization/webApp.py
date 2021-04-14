"""Streamlit web app to display Icestupa class object
"""

# External modules
import streamlit as st
import pandas as pd
import sys
from datetime import datetime
import os
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


# SETTING PAGE CONFIG TO WIDE MODE
air_logo = os.path.join(dirname, "src/visualization/AIR_logo_circle.png")
st.set_page_config(
    layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="Icestupa",  # String or None. Strings get appended with "• Streamlit".
    # page_icon=None,  # String, anything supported by st.image, or None.
    page_icon=air_logo,  # String, anything supported by st.image, or None.
)


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
        ("Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020", "Gangles 2021"),
    )
    trigger = st.sidebar.radio(
        "Select Fountain control", ("Manual", "Weather", "Temperature", "None")
    )
    SITE, FOUNTAIN, FOLDER = config(location, trigger=trigger)

    icestupa = Icestupa(SITE, FOUNTAIN, FOLDER)

    try:
        icestupa.read_output()
        df_in = icestupa.df
        (
            input_cols,
            input_vars,
            output_cols,
            output_vars,
            derived_cols,
            derived_vars,
        ) = vars(df_in)
        df_in = df_in[df_in.columns.drop(list(df_in.filter(regex="Unnamed")))]
        df_in = df_in.set_index("When")
        df = df_in
        input_folder = os.path.join(dirname, "data/" + SITE["name"] + "/interim/")
        output_folder = os.path.join(dirname, "data/" + SITE["name"] + "/processed/")
        row1_1, row1_2 = st.beta_columns((2, 5))

        with row1_1:
            st.image(air_logo, width=160)

        with row1_2:
            st.markdown(
                """
            # Artificial Ice Reservoirs of **_%s_**

            """
                % location
            )
            st.text("")
            visualize = [
                "Validation",
                "Timelapse",
                "Data Overview",
                "Input",
                "Output",
                "Derived",
            ]
            display = st.multiselect(
                "Choose type of visualization below:",
                options=(visualize),
                default=["Validation", "Timelapse"],
                # default=["Validation", "Timelapse"],
            )

            if trigger == "None":
                st.write(
                    """
                Fountain was always kept on until **%s**
                """
                    % (icestupa.fountain_off_date.date())
                )
            if trigger == "Manual":
                st.write(
                    """
                Fountain was controlled **%s** until **%s**
                """
                    % (trigger + "ly", (icestupa.fountain_off_date.date()))
                )
            if trigger == "Temperature":
                st.write(
                    """
                Fountain was switched on/off after sunset when temperature was below **%s** until **%s**
                """
                    % (icestupa.crit_temp, (icestupa.fountain_off_date.date()))
                )
            if trigger == "Weather":
                st.write(
                    """
                Fountain was switched on/off whenever surface energy balance was negative/positive respectively until **%s**
                """
                    % (icestupa.fountain_off_date.date())
                )

        st.sidebar.write("### Map")
        lat = SITE["latitude"]
        lon = SITE["longitude"]
        map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.sidebar.map(map_data, zoom=10)
        st.sidebar.write(
            """
        ### More Info
        [![Star](https://img.shields.io/github/stars/Gayashiva/air_model?logo=github&style=social)](https://gitHub.com/Gayashiva/air_model)
        &nbsp[![Follow](https://img.shields.io/twitter/follow/know_just_ice?style=social)](https://www.twitter.com/know_just_ice)
        """
        )

        row2_1, row2_2 = st.beta_columns((1, 1))
        with row2_1:
            Efficiency = (
                (icestupa.df["meltwater"].iloc[-1] + icestupa.df["ice"].iloc[-1])
                / icestupa.df["input"].iloc[-1]
                * 100
            )
            Duration = icestupa.df.index[-1] * icestupa.TIME_STEP / (60 * 60 * 24)
            st.markdown(
                """
            | Fountain attributes | Value |
            | --- | --- |
            | Active from | %s |
            | Last active on | %s |
            | Aperture diameter | %.2f $mm$|
            | Initial height | %s $m$ |
            """
                % (
                    icestupa.start_date.date(),
                    icestupa.fountain_off_date.date(),
                    icestupa.dia_f,
                    icestupa.h_f,
                )
            )

        with row2_2:
            st.markdown(
                """
            | Icestupa properties | Model output |
            | --- | --- |
            | Maximum Ice Volume | %.2f $m^{3}$|
            | Storage Efficiency | %.2f percent |
            | Meltwater released | %.2f $l$ |
            | Total Precipitation | %.2f $kg$ |
            | Model duration | %.2f days |
            """
                % (
                    icestupa.df["iceV"].max(),
                    Efficiency,
                    icestupa.df["meltwater"].iloc[-1],
                    icestupa.df["ppt"].sum(),
                    Duration,
                )
            )

        if not (display):
            st.error("Please select at least one visualization.")
        else:
            if "Validation" in display:
                st.write("## Validation")
                path = (
                    output_folder
                    + "paper_figures/Vol_Validation_"
                    + icestupa.trigger
                    + ".jpg"
                )
                st.image(path)

                if SITE["name"] in ["guttannen21", "guttannen20"]:
                    path = (
                        output_folder
                        + "paper_figures/Temp_Validation_"
                        + icestupa.trigger
                        + ".jpg"
                    )
                    st.image(path)

            if "Timelapse" in display:
                st.write("## Timelapse")
                if location == "Schwarzsee 2019":
                    url = "https://youtu.be/GhljRBGpxMg"
                    st.video(url)
                elif location == "Guttannen 2021":
                    url = "https://youtu.be/DBHoL1Z7H6U"
                    st.video(url)
                elif location == "Guttannen 2020":
                    st.error("No Timelapse recorded")

            if "Data Overview" in display:
                st.write("## Input variables")
                st.image(
                    output_folder + "paper_figures/Model_Input_" + trigger + ".jpg"
                )
                st.write(
                    """
                Measurements at the AWS of %s were used as main model input
                data in 15 minute frequency.  Incoming shortwave and longwave radiation
                were obtained from ERA5 reanalysis dataset. Several data gaps
                and errors were also filled from the ERA5 dataset (shaded regions).  
                """
                    % (icestupa.name)
                )
                st.write("## Output variables")
                st.image(
                    output_folder + "paper_figures/Model_Output_" + trigger + ".jpg"
                )
                st.write(
                    """
                (a) Fountain discharge (b) energy flux components, (c) mass flux components (d)
                surface area and (e) volume of the Icestupa in daily time steps. qSW is the net
                shortwave radiation; qLW is the net longwave radiation; qL and qS are the
                turbulent latent and sensible heat fluxes. qF represents the interactions of
                the ice-water boundary during fountain on time steps. qG quantifies the heat
                conduction process between the Icestupa surface layer and the ice body.
                """
                )

            if "Input" in display:
                st.write("## Input variables")
                variable1 = st.multiselect(
                    "Choose",
                    options=(input_cols),
                    default=["Discharge", "Temperature"],
                    # default=["Temperature"],
                )
                if not (variable1):
                    st.error("Please select at least one variable.")
                else:
                    variable_in = [
                        input_vars[input_cols.index(item)] for item in variable1
                    ]
                    variable = variable_in
                    for v in variable:

                        meta = get_parameter_metadata(v)
                        st.header("%s" % (meta["name"] + " " + meta["units"]))
                        st.line_chart(df[v])

            if "Output" in display:
                st.write("## Output variables")

                variable2 = st.multiselect(
                    "Choose",
                    options=(output_cols),
                    default=["Meltwater"],
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

            if "Derived" in display:
                st.write("## Derived variables")
                variable3 = st.multiselect(
                    "Choose",
                    options=(derived_cols),
                    default=["Albedo"],
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
                        st.header("%s" % (meta["name"] + " " + meta["units"]))
                        st.line_chart(df[v])

    except FileNotFoundError:
        st.error(
            "Sorry, yet to produce relevant outputs for fountain trigger %s. Try another fountain trigger."
            % icestupa.trigger
        )
