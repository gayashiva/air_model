"""Location specific settings used to initialise icestupa object
"""

# External modules
import pandas as pd
from datetime import datetime
import logging
import os, sys
import numpy as np

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)

# Spammers
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)

def config(location="guttannen21", spray="man"):

    if location == "Guttannen 2022" or location == "guttannen22":

        SITE = dict(
            name="guttannen22",
            alt=1047.6,
            coords=[46.65549,8.29149],
            # cld=0.5,
            # Calibrated values
            DX=45e-03,  # Surface layer thickness [m]
        )

        if spray == "auto":
            add= dict(
                start_date=datetime(2021, 12, 3, 8),
                fountain_off_date=datetime(2022, 1, 27),
                dis_crit = 1,
                dis_max= 13,
                h_i = 0.13, #Initialise ice height at start
                # R_F=3,  # First drone rad
            # perimeter=35, # on Jan 28
            )

            f_heights = [
                {"time": add["start_date"], "h_f": 3},
                {"time": datetime(2022, 12, 23, 16), "h_f": 4},
            ]

        if spray == "man":
            add = dict(
                start_date=datetime(2021, 12, 8, 14),
                expiry_date=datetime(2022, 1, 27),
                fountain_off_date=datetime(2022, 1, 27),
                h_i = 0.13, #Initialise ice height at start
            )

            f_heights = [
                {"time": add["start_date"], "h_f": 3.7},
                {"time": datetime(2022, 12, 23, 16), "h_f": 4.7},
                {"time": datetime(2022, 2, 12, 16), "h_f": 5.7},
            ]
        SITE = dict(SITE, **add)

    if location == "Guttannen 2021" or location == "guttannen21":

        SITE = dict(
            name="guttannen21",
            alt=1047.6,
            coords=[46.65549,8.29149],
            # cld=0.5,
            # R_F=4.3,  # Fountain mean discharge
            # R_F=5.4,  # First drone rad
            # h_f=5,
            # perimeter=45, # on Feb 11

            # Calibrated values
            DX=45e-03,  # Surface layer thickness [m]
        )

        if spray == "auto":
            add= dict(
                start_date=datetime(2020, 11, 22, 15),
                fountain_off_date=datetime(2021, 2, 10, 1),
                dis_crit = 1,
                dis_max= 13,
            )

        if spray == "man":
            add = dict(
                start_date=datetime(2020, 11, 22, 15),
                expiry_date=datetime(2021, 5, 10, 1),
                fountain_off_date=datetime(2021, 2, 20, 10),
                D_F=7.5,  # Fountain mean discharge
            )
        SITE = dict(SITE, **add)

        f_heights = [
            {"time": SITE["start_date"], "h_f": 2.68},
            {"time": datetime(2020, 12, 30, 16), "h_f": 3.75},
            {"time": datetime(2021, 1, 7, 16), "h_f": 4.68},
            {"time": datetime(2021, 1, 11, 16), "h_f": 5.68},
        ]

    if location == "Guttannen 2020" or location == "guttannen20":

        SITE = dict(
            name="guttannen20",
            alt=1047.6,
            coords=[46.65549,8.29149],
            # R_F=6.68,  # First drone rad
            # cld=0.5,
            # h_f=3,
            # perimeter=28, # on 24 Jan

            # Calibrated values
            DX=45e-03,  # Surface layer thickness [m]
        )

        if spray == "auto":
            add= dict(
                start_date=datetime(2020, 1, 3, 16),
                fountain_off_date=datetime(2020, 3, 6, 12),
                dis_crit = 1,
                dis_max= 13,
            )
        if spray == "man":
            add = dict(
                start_date=datetime(2020, 1, 3, 16),
                expiry_date=datetime(2020, 4, 6, 12),
                fountain_off_date=datetime(2020, 3, 8, 9),  # Image shows Dani switched off at 8th Mar 10 am
                D_F=7.5,  # Fountain mean discharge
            )
        SITE = dict(SITE, **add)

        f_heights = [
            {"time": SITE["start_date"], "h_f": 2.5},
            {"time": datetime(2020, 1, 24, 12), "h_f": 3.5},
            {"time": datetime(2020, 2, 5, 19), "h_f": 2.5},
        ]

    if location == "Gangles 2021" or location == "gangles21":

        SITE = dict(
            name="gangles21",
            alt=4009,
            coords=[34.216638,77.606949],
            # h_f=9,
            # cld=0.1,
            # R_F=9.05,  # First drone rad
            # perimeter=82.3, # On 3 Mar

            # Calibrated values
            DX=65e-03,  # Surface layer thickness [m]
        )

        f_heights = [
            {"time": SITE["start_date"], "h_f": 5},
            {"time": datetime(2021, 1, 22, 16), "h_f": 9},
        ]

        if spray == "auto":
            add= dict(
                start_date=datetime(2021, 1, 18),
                fountain_off_date=datetime(2021, 4, 10),
                dis_crit = 1,
                dis_max= 60,
            )
        if spray == "man":
            add = dict(
                start_date=datetime(2021, 1, 18),
                expiry_date=datetime(2021, 6, 20),
                fountain_off_date=datetime(2021, 3, 10, 18),
                D_F=60,  # FOUNTAIN min discharge
            )
        SITE = dict(SITE, **add)

    # Define directory structure
    FOLDER = dict(
        raw="data/" + SITE["name"] + "/raw/",
        input="data/" + SITE["name"] + "/interim/",
        output="data/" + SITE["name"] + "/processed/",
        sim="data/" + SITE["name"] + "/processed/simulations/",
        fig="data/" + SITE["name"] + "/figs/",
    )
    df_h = pd.DataFrame(f_heights)

    return SITE, FOLDER
