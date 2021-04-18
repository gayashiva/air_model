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
from src.utils.settings import config
from src.utils import setup_logger


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

    st.sidebar.write("### Select Icestupa")
    location = st.sidebar.radio(
        "built at",
        ("Guttannen 2021", "Gangles 2021", "Guttannen 2020",  "Schwarzsee 2019"),
        # ("Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020", "Gangles 2021"),
        # ("Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020"),
    )

    # st.sidebar.write("### Fountain")
    # trigger = st.sidebar.radio(
    #     "controlled by", ("Field staff", "Weather", "Temperature", "None")
    # )
    # if trigger == "Field staff":
    #     trigger = "Manual"


    if location in ['Gangles 2021']:
        trigger = "None"
    else:
        trigger = "Manual"

    SITE, FOUNTAIN, FOLDER = config(location, trigger=trigger)


    icestupa = Icestupa(location, trigger)

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
            # **_%s_** Ice Reservoir

            """
                % location.split()[0]
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
                Fountain was switched on/off after sunset when temperature was below **%s**
                """
                    % icestupa.crit_temp
                )
            if trigger == "Weather":
                st.write(
                    """
                Fountain discharge was set based on magnitude of surface energy balance.
                """
                )
            visualize = [
                "Timelapse",
                "Validation",
                "Data Overview",
                "Input",
                "Output",
                # "Derived",
            ]
            display = st.multiselect(
                "Choose type of visualization below:",
                options=(visualize),
                default=["Validation"],
                # default=["Validation", "Timelapse"],
            )

        st.sidebar.write("### Map")
        lat = SITE["latitude"]
        lon = SITE["longitude"]
        map_data = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.sidebar.map(map_data, zoom=10)

        st.sidebar.write(
            """
        ### About
        Several villages in the arid high Himalayas have been constructing
        [artificial ice
        reservoirs](https://www.thethirdpole.net/en/climate/the-glacier-marriages-in-pakistans-high-himalayas/)
        to meet their farming water demand in early spring. With the invention of
        [icestupas](https://www.youtube.com/watch?v=2xuBvI98-n4&t=2s) this
        practice of storing water as ice now shows great potential over
        traditional water storage techniques. It doesn't need any energy to
        construct and the materials needed like pipelines and fountain are
        often already available. The only major limitation though is where this
        technology can be applied, since it requires certain favourable weather
        conditions in order to freeze the available water.  In order to identify such suitable regions, we developed a
        physical model that takes weather conditions and water availability as
        input and estimates the amount of meltwater expected.

        In the winters of 2019, 2020 and 2021, several scientific
        icestupas were built in India and Switzerland to calibrate and
        validate this physical model. Here we present the model results.

        [Abstract](https://www.unifr.ch/geo/cryosphere/en/projects/smd4gc/artificial-ice-reservoirs.html)

        [![Star](https://img.shields.io/github/stars/Gayashiva/air_model?logo=github&style=social)](https://gitHub.com/Gayashiva/air_model)

        [![Follow](https://img.shields.io/twitter/follow/know_just_ice?style=social)](https://www.twitter.com/know_just_ice)
        """
        )

        row2_1, row2_2 = st.beta_columns((1, 1))
        with row2_1:
            f_mean = icestupa.df.Discharge.replace(0, np.nan).mean()
            f_efficiency = 100 - (
                (
                    icestupa.df["unfrozen_water"].iloc[-1]
                    / (icestupa.df["Discharge"].sum() * icestupa.TIME_STEP / 60)
                    * 100
                )
            )
            Duration = icestupa.df.index[-1] * icestupa.TIME_STEP / (60 * 60 * 24)
            st.markdown(
                """
            | Fountain | Estimation |
            | --- | --- |
            | Mean discharge | %.1f $l/min$|
            | Water frozen| %.1f percent |
            | Water sprayed| %.0f $m^3$ |
            | Used for | %.0f hours |
            """
                % (
                    f_mean,
                    f_efficiency,
                    icestupa.df.Discharge.sum() * icestupa.TIME_STEP / (60 * 1000),
                    icestupa.df.Discharge.astype(bool).sum(axis=0)
                    * icestupa.TIME_STEP
                    / 3600,
                )
            )

        with row2_2:
            st.markdown(
                """
            | Icestupa| Estimation |
            | --- | --- |
            | Max Ice Volume | %.1f $m^{3}$|
            | Meltwater released | %.0f $kg$ |
            | Ice remaining | %.0f $kg$ |
            | Vapour loss | %.0f $kg$ |
            """
                % (
                    icestupa.df["iceV"].max(),
                    icestupa.df["meltwater"].iloc[-1],
                    icestupa.df["ice"].iloc[-1],
                    icestupa.df["vapour"].iloc[-1],
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

                # if SITE["name"] in ["guttannen21", "guttannen20"]:
                #     path = (
                #         output_folder
                #         + "paper_figures/Temp_Validation_"
                #         + icestupa.trigger
                #         + ".jpg"
                #     )
                #     st.image(path)

            if "Timelapse" in display:
                st.write("## Timelapse")
                if location == "Schwarzsee 2019":
                    url = "https://youtu.be/GhljRBGpxMg"
                    st.video(url)
                elif location == "Guttannen 2021":
                    url = "https://youtu.be/DBHoL1Z7H6U"
                    st.video(url)
                elif location == "Guttannen 2020":
                    url = "https://youtu.be/kcrvhU20OOE"
                    st.video(url)
                elif location == "Gangles 2021":
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
