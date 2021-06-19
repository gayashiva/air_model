"""Streamlit web app to display Icestupa class object
"""

# External modules
import streamlit as st
import pandas as pd
import sys
from datetime import datetime, timedelta
import os
import numpy as np
import re
import base64
import logging
import coloredlogs
from pathlib import Path

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger


# SETTING PAGE CONFIG TO WIDE MODE
air_logo = os.path.join(dirname, "src/visualization/logos/AIR_logo_circle.png")
st.set_page_config(
    layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
    initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
    page_title="Icestupa",  # String or None. Strings get appended with "• Streamlit".
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

    st.sidebar.markdown(
        """
    # Ice Reservoir

    """
    )

    location = st.sidebar.radio(
        "built at",
        # ( "Guttannen 2021","Gangles 2021", "Diavolezza 2021","Guttannen 2020", "Schwarzsee 2019"),
        # ("Guttannen 2021", "Gangles 2021", "Guttannen 2020", "Schwarzsee 2019"),
        ("Guttannen 2021", "Gangles 2021", "Guttannen 2020", "Schwarzsee 2019"),
        # ("Guttannen 2021", "Guttannen 2020", "Schwarzsee 2019"),
    )

    # location = "Gangles 2021"
    trigger = "Manual"

    SITE, FOLDER = config(location)

    icestupa = Icestupa(location)

    icestupa.read_input()
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
        # **_%s_** Icestupa

        """
            % location.split()[0]
        )
        visualize = [
            "Timelapse",
            "Validation",
            "Data Overview",
            "Input",
            "Output",
            "Derived",
        ]
        display = st.multiselect(
            "Choose type of visualization below:",
            options=(visualize),
            default=["Validation"],
            # default=["Validation", "Timelapse"],
        )
        intro_markdown = Path("src/visualization/intro.md").read_text()
        st.markdown(intro_markdown, unsafe_allow_html=True)

    st.markdown("---")
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
    often already available to a farmer. One major limitation though is where this
    technology can be applied, since it requires certain favourable weather
    conditions in order to freeze the available water.  In order to identify such suitable regions, we developed a
    physical model that takes weather conditions and water availability as
    input and estimates the amount of meltwater expected.

    [![Follow](https://img.shields.io/twitter/follow/know_just_ice?style=social)](https://www.twitter.com/know_just_ice)
    """
    )

    st.sidebar.write(
        """
        ### Partners
        """
    )
    row2_1, row2_2, row2_3 = st.sidebar.beta_columns((1, 1, 1))
    row3_1, row3_2 = st.beta_columns((1, 1))
    with row2_1:
        st.image(
            "src/visualization/logos/unifr.png",
            caption="UniFR",
            use_column_width=True,
        )
        st.markdown(" ")
        st.image(
            "src/visualization/logos/GA.png",
            caption="GlaciersAlive",
            use_column_width=True,
        )
        st.markdown(" ")
        st.image(
            "src/visualization/logos/ng-logo.png",
            # caption="GlaciersAlive",
            use_column_width=True,
        )
    with row2_2:
        st.image(
            "src/visualization/logos/HIAL-logo.png",
            caption="HIAL",
            use_column_width=True,
        )
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.image(
            "src/visualization/logos/logo-schwarzsee.png",
            caption="Schwarzsee Tourism",
            use_column_width=True,
        )
        st.markdown(" ")
        st.markdown(" ")
        st.image(
            "src/visualization/logos/dfrobot.png",
            # caption="GlaciersAlive",
            use_column_width=True,
        )
    with row2_3:
        st.image(
            "src/visualization/logos/guttannen-bewegt.png",
            caption="Guttannen Moves",
            use_column_width=True,
        )
        st.markdown(" ")
        st.markdown(" ")
        st.markdown(" ")
        st.image(
            "src/visualization/logos/Logo-Swiss-Polar-Institute.png",
            # caption="Swiss Polar Institute",
            use_column_width=True,
        )

    with row3_1:
        f_mean = icestupa.df.Discharge.replace(0, np.nan).mean()
        f_efficiency = 100 - (
            (
                icestupa.df["unfrozen_water"].iloc[-1]
                / (icestupa.df["Discharge"].sum() * icestupa.DT / 60)
                * 100
            )
        )
        Duration = icestupa.df.index[-1] * icestupa.DT / (60 * 60 * 24)
        st.markdown(
            """
        | Fountain | Estimation |
        | --- | --- |
        | Mean discharge | %.1f $l/min$|
        | Spray Radius | %.1f $m$|
        | Water sprayed| %.0f $m^3$ |
        | Storage Efficiency | %.0f $percent$ |
        """
            % (
                f_mean,
                icestupa.r_spray,
                icestupa.df.Discharge.sum() * icestupa.DT / (60 * 1000) ,
                f_efficiency,
            )
        )

    with row3_2:
        st.markdown(
            """
        | Icestupa| Estimation |
        | --- | --- |
        | Max Ice Volume | %.1f $m^{3}$|
        | Meltwater released | %.0f $kg$ |
        | Vapour loss | %.0f $kg$ |
        | Storage Duration | %.0f $days$ |
        """
            % (
                icestupa.df["iceV"].max(),
                icestupa.df["meltwater"].iloc[-1],
                icestupa.df["vapour"].iloc[-1],
                Duration,
            )
        )

    st.markdown("---")
    if not (display):
        st.error("Please select at least one visualization.")
    else:
        if "Validation" in display:
            df_c = pd.read_hdf(icestupa.input + "model_input_" + icestupa.trigger + ".h5", "df_c")

            # df_c = pd.read_hdf(icestupa.input + "model_input_" + icestupa.trigger + ".h5", "df_c")
            df_c = df_c.set_index("When")
            icestupa.df= icestupa.df.set_index("When")
            tol = pd.Timedelta('1T')
            df = pd.merge_asof(left=icestupa.df,right=df_c,right_index=True,left_index=True,direction='nearest',tolerance=tol)

            ctr = 0
            while (df[df.DroneV.notnull()].shape[0]) == 0 and ctr !=4:
                tol += pd.Timedelta('15T')
                logger.error("Timedelta increase as shape %s" %(df[df.DroneV.notnull()].shape[0]))
                df = pd.merge_asof(left=icestupa.df,right=df_c,right_index=True,left_index=True,direction='nearest',tolerance=tol)
                ctr+=1


            rmse_V = (((df.DroneV - df.iceV) ** 2).mean() ** .5)
            corr_V = df['DroneV'].corr(df['iceV'])
            

            if icestupa.name in ["guttannen21", "guttannen20"]:
                df_cam = pd.read_hdf(icestupa.input + "model_input_" + icestupa.trigger + ".h5", "df_cam")
                df = pd.merge_asof(left=icestupa.df,right=df_cam,right_index=True,left_index=True,direction='nearest',tolerance=tol)
                rmse_T = (((df.cam_temp - df.T_s) ** 2).mean() ** .5)
                corr_T = df['cam_temp'].corr(df['T_s'])
            else:
                rmse_T = 0
                corr_T = 0

            st.write("## Validation")
            path = (
                output_folder
                + "paper_figures/Vol_Validation_"
                + icestupa.trigger
                + ".jpg"
            )
            st.image(path)
            st.write(
                """
            Correlation of modelled with measured ice volume was **%.2f** and RMSE was **%.2f** $m^3$ 
            """
                % (corr_V, rmse_V)
            )

            if SITE["name"] in ["guttannen21", "guttannen20"]:
                path = (
                    output_folder
                    + "paper_figures/Temp_Validation_"
                    + icestupa.trigger
                    + ".jpg"
                )
                st.image(path)
                st.write(
                    """
                Correlation of modelled with measured surface temperature was **%.2f** and RMSE was **%.2f** C
                """
                    % (corr_T, rmse_T)
                )

        if "Timelapse" in display:
            st.write("## Timelapse")
            if location == "Schwarzsee 2019":
                url = "https://youtu.be/GhljRBGpxMg"
                st.video(url)
            elif location == "Guttannen 2021":
                url = "https://www.youtube.com/watch?v=kXi4abO4YVM"
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
                    row4_1, row4_2 = st.beta_columns((2, 5))
                    with row4_1:
                        st.write(df[v].describe())
                    with row4_2:
                        st.line_chart(df[v], use_container_width=True)

        if "Output" in display:
            st.write("## Output variables")

            variable2 = st.multiselect(
                "Choose",
                options=(output_cols),
                default=["Discharge Runoff"],
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
                    row5_1, row5_2 = st.beta_columns((2, 5))
                    with row5_1:
                        st.write(df[v].describe())
                    with row5_2:
                        st.line_chart(df[v], use_container_width=True)

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
                    row6_1, row6_2 = st.beta_columns((2, 5))
                    with row6_1:
                        st.write(df[v].describe())
                    with row6_2:
                        st.line_chart(df[v], use_container_width=True)
