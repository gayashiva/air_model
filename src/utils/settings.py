"""loc specific settings used to initialise icestupa object
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
logging.getLogger('PIL').setLevel(logging.CRITICAL)

# Module logger
logger = logging.getLogger("__main__")

def config(loc="guttannen21", spray=None):
    logger.warning(f"Site {loc} with scheduler {spray}")

    if loc== "Guttannen 2022" or loc == "guttannen22":
        SITE = dict(
            name="guttannen22",
            alt=1047.6,
            coords=[46.65549,8.29149],
            start_date=datetime(2021, 12, 3, 12),
            expiry_date =datetime(2022, 3, 3),
            h_dome = 0.13, #Initialise ice height at start
            utc = 1, #Initialise ice height at start
            cld=0.5,
            # Calibrated values
            DX=45e-03,  # Surface layer thickness [m]
        )

        if spray != None:
            if spray.split('_')[0] == "scheduled":
                if spray.split('_')[1] in ["wue", "icv", "field"]:
                    add= dict(
                        fountain_off_date=datetime(2022, 3, 3),
                        R_F = 4,
                        dis_crit = 2,
                    )
            if spray.split('_')[0] == "unscheduled":
                if spray.split('_')[1] == "field":
                    add= dict(
                        fountain_off_date=datetime(2022, 2, 17, 6),
                        f_heights = [
                            {"time": datetime(2021, 12, 8, 14), "h_f": 3.7},
                            {"time": datetime(2021, 12, 23, 16), "h_f": 4.7},
                            {"time": datetime(2022, 2, 12, 16), "h_f": 5.7},
                        ],
                    )
            SITE = dict(SITE, **add)

    if loc == "Gangles 2021" or loc == "gangles21":

        SITE = dict(
            name="gangles21",
            start_date=datetime(2021, 1, 18),
            expiry_date=datetime(2021, 4, 10),
            alt=4009,
            coords=[34.216638,77.606949],
            utc=4.5,
            cld=0.1,
            # h_f=9,
            # R_F=9.05,  # First drone rad
            # perimeter=82.3, # On 3 Mar

            # Calibrated values
            DX=65e-03,  # Surface layer thickness [m]
        )

        if spray != None:
            if spray.split('_')[0] == "scheduled":
                if spray.split('_')[1] in ["wue", "icv"]:
                    add= dict(
                        fountain_off_date=datetime(2021, 4, 10),
                        R_F=10,
                        dis_crit = 1,
                        # dis_max= 60,
                        # R_F = 10,
                    )

            if spray.split('_')[0] == "unscheduled":
                if spray.split('_')[1] in ["field"]:
                    add = dict(
                        fountain_off_date=datetime(2021, 3, 10, 18),
                        D_F=60,  # FOUNTAIN infinite water
                        # dis_max=60,  # FOUNTAIN min discharge
                        f_heights = [
                            {"time": datetime(2021, 1, 18), "h_f": 5},
                            {"time": datetime(2021, 1, 22, 16), "h_f": 9},
                        ],
                    )

            SITE = dict(SITE, **add)

    if loc == "Guttannen 2021" or loc == "guttannen21":
        SITE = dict(
            name="guttannen21",
            alt=1047.6,
            coords=[46.65549,8.29149],
            utc=1,
            start_date=datetime(2020, 11, 22, 15),
            expiry_date=datetime(2021, 5, 10, 1),
            cld=0.5,
            # R_F=4.3,  # Fountain mean discharge
            # R_F=5.4,  # First drone rad
            # h_f=5,
            # perimeter=45, # on Feb 11

            # Calibrated values
            DX=45e-03,  # Surface layer thickness [m]
        )

        if spray != None:
            if spray.split('_')[0] == "scheduled":
                if spray.split('_')[1] in ["wue", "icv"]:
                    add= dict(
                        dis_crit = 2,
                        R_F = 6.9,
                    )
                add["fountain_off_date"] = add["expiry_date"]

            if spray.split('_')[0] == "unscheduled":
                if spray.split('_')[1] == "field":
                    add = dict(
                        fountain_off_date=datetime(2021, 2, 20, 10),
                        dis_max= 18,
                        D_F=7.5,  # Fountain mean discharge
                        f_heights = [
                            {"time": datetime(2020, 11, 22, 15), "h_f": 2.68},
                            {"time": datetime(2020, 12, 30, 16), "h_f": 3.75},
                            {"time": datetime(2021, 1, 7, 16), "h_f": 4.68},
                            {"time": datetime(2021, 1, 11, 16), "h_f": 5.68},
                        ],
                    )

            SITE = dict(SITE, **add)


    if loc == "Guttannen 2020" or loc == "guttannen20":

        SITE = dict(
            name="guttannen20",
            alt=1047.6,
            coords=[46.65549,8.29149],
            # R_F=6.68,  # First drone rad
            cld=0.5,
            # h_f=3,
            # perimeter=28, # on 24 Jan

            # Calibrated values
            DX=45e-03,  # Surface layer thickness [m]
        )

        if spray != None:

            if spray == "dynamic" or spray == "static":
                add= dict(
                    start_date=datetime(2020, 1, 3, 16),
                    fountain_off_date=datetime(2020, 3, 6, 12),
                    dis_crit = 1,
                    dis_max= 11,
                    # R_F = 7,
                )
                add["expiry_date"] = add["fountain_off_date"]

            if spray == "manual":
                add = dict(
                    start_date=datetime(2020, 1, 3, 16),
                    expiry_date=datetime(2020, 4, 6, 12),
                    fountain_off_date=datetime(2020, 3, 8, 9),  # Image shows Dani switched off at 8th Mar 10 am
                    dis_max= 11,
                    D_F=7.5,  # Fountain mean discharge
                    f_heights = [
                        {"time": datetime(2020, 1, 3, 16), "h_f": 2.5},
                        {"time": datetime(2020, 1, 24, 12), "h_f": 3.5},
                        {"time": datetime(2020, 2, 5, 19), "h_f": 2.5},
                    ],
                )

            SITE = dict(SITE, **add)



    # Define directory structure
    if spray != None:
        FOLDER = dict(
            raw="data/" + SITE["name"] + "/raw/",
            input="data/" + SITE["name"] + "/interim/",
            input_sim="data/" + SITE["name"] + "/interim/" + spray.split('_')[0] + "/" + spray.split('_')[1]+ "/",
            output="data/" + SITE["name"] + "/processed/"+ spray.split('_')[0] + "/" + spray.split('_')[1]+ "/",
            fig="data/" + SITE["name"] + "/figs/"+ spray.split('_')[0] + "/" + spray.split('_')[1]+ "/",
        )
    else:
        FOLDER = dict(
            raw="data/" + SITE["name"] + "/raw/",
            input="data/" + SITE["name"] + "/interim/",
            output="data/" + SITE["name"] + "/processed/",
            fig="data/" + SITE["name"] + "/figs/",
        )


    return SITE, FOLDER
