"""Location specific settings used to initialise icestupa object
"""

# External modules
import pandas as pd
from datetime import datetime
import logging
import os, sys
import numpy as np

# Module logger
logger = logging.getLogger(__name__)

# Spammers
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)


def config(location="guttannen21"):

    logger.info("Location is %s" % (location))

    if location == "Guttannen 2021" or location == "guttannen21":

        SITE = dict(
            name="guttannen21",
            start_date=datetime(2020, 11, 22, 15),
            # end_date=datetime(2021, 5, 10, 1),
            melt_out=datetime(2021, 5, 10, 1),
            fountain_off_date=datetime(2021, 2, 20, 10),
            D_F=7.5,  # FOUNTAIN min discharge
            T_F=3,  # FOUNTAIN min discharge
            utc_offset=2,
            latitude=46.649999,
            longitude=8.283333,
            H_AWS=2,
            # R_F=10.2,
            SA_corr=1.2,
            Z=0.001,
            # perimeter=45, # on Feb 11
            # DX= 50e-03,
            # Z= 5e-03,
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 2.5},
            {"When": datetime(2020, 12, 30, 16), "h_f": 3.5},
            {"When": datetime(2021, 1, 7, 16), "h_f": 5.5},
            {"When": datetime(2021, 1, 11, 16), "h_f": 4.5},
        ]

    if location == "Guttannen 2020" or location == "guttannen20":

        SITE = dict(
            name="guttannen20",
            start_date=datetime(2020, 1, 3, 16),
            # end_date=datetime(2020, 4, 6, 12),
            melt_out=datetime(2020, 4, 6, 12),
            fountain_off_date=datetime(
                2020, 3, 8, 9
            ),  # Image shows Dani switched off at 8th Mar 10 am
            D_F=7.5,  # FOUNTAIN min discharge
            T_F=3,  # FOUNTAIN min discharge
            utc_offset=2,
            latitude=46.649999,
            longitude=8.283333,
            H_AWS=2,
            # perimeter=28, # on 24 Jan
            # DX= 50e-03,
            # Z= 5e-03,
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 2.5},
            {"When": datetime(2020, 1, 24, 12), "h_f": 3.5},
            {"When": datetime(2020, 2, 5, 19), "h_f": 2.5},
        ]

    if location == "Gangles 2021" or location == "gangles21":

        SITE = dict(
            name="gangles21",
            start_date=datetime(2021, 1, 18),
            # end_date=datetime(2021, 7, 8),
            melt_out=datetime(2021, 6, 20),  # Norboo observed
            fountain_off_date=datetime(2021, 3, 10, 18),
            D_F=60,  # FOUNTAIN min discharge
            T_F=1,  # FOUNTAIN min discharge
            utc_offset=5.5,
            longitude=77.606949,
            latitude=34.216638,
            H_AWS=2,
            diffuse_fraction=0,
            SA_corr=1.5,
            # perimeter=82.3, # On 3 Mar
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5},
            # {"When": datetime(2021, 1, 22, 16), "h_f": 9},
        ]

    if location == "Schwarzsee 2019" or location == "schwarzsee19":
        SITE = dict(
            name="schwarzsee19",
            start_date=datetime(2019, 1, 30, 17),
            # end_date=datetime(2019, 3, 17),
            melt_out=datetime(2019, 3, 10, 19),
            fountain_off_date=datetime(2019, 2, 16, 10),
            T_F=1,  # FOUNTAIN min discharge
            utc_offset=1,
            longitude=7.297543,
            latitude=46.693723,
            R_F=1.233,
            H_AWS=2,
            # discharge=3.58,  # FOUNTAIN on mean discharge from field
            # dia_f=0.0056,  # FOUNTAIN aperture diameter
            # DX= 50e-03,
            # Z= 5e-03,
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 1.35},
        ]

    if location == "Phortse 2020" or location == "phortse20":

        SITE = dict(
            name="phortse20",
            start_date=datetime(2019, 12, 1),
            melt_out=datetime(2020, 2, 1),
            # melt_out=datetime(2021, 5, 10, 1),
            fountain_off_date=datetime(2020, 2, 1),
            D_F=60,
            T_F=3,
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            H_AWS=2,
            SA_corr=1.2,
            Z=0.001,
            R_F=10,
            diffuse_fraction=0,
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5},
        ]

    # Define directory structure
    FOLDER = dict(
        raw="data/" + SITE["name"] + "/raw/",
        input="data/" + SITE["name"] + "/interim/",
        output="data/" + SITE["name"] + "/processed/",
        sim="data/" + SITE["name"] + "/processed/simulations/",
    )
    df_h = pd.DataFrame(data_h)

    return SITE, FOLDER
