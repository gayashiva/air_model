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
            start_date=datetime(2020, 11, 22,15),
            end_date=datetime(2021, 5, 10, 1),
            fountain_off_date=datetime(2021, 2, 20,10),
            D_F=7.5,  # FOUNTAIN min discharge
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            H_AWS = 2,
            # perimeter=45, # on Feb 11
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
            end_date=datetime(2020, 4, 6, 12),
            fountain_off_date=datetime(2020, 3, 8,9), # Image shows Dani switched off at 8th Mar 10 am
            D_F=7.5,  # FOUNTAIN min discharge
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            H_AWS = 2,
            # perimeter=28, # on 24 Jan
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
            end_date=datetime(2021, 6, 11),
            fountain_off_date=datetime(2021, 3, 10, 18),
            D_F=5,  # FOUNTAIN min discharge
            utc_offset=5.5,
            longitude=77.606949,
            latitude=34.216638,
            H_AWS = 2,
            diffuse_fraction = 0,
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
            end_date=datetime(2019, 3, 10, 19),
            fountain_off_date=datetime(2019, 2, 16, 10),
            utc_offset=1,
            longitude=7.297543,
            latitude=46.693723,
            r_spray = 1.233,
            H_AWS = 2,
            # discharge=3.58,  # FOUNTAIN on mean discharge from field
            # dia_f=0.0056,  # FOUNTAIN aperture diameter
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 1.35},
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


if __name__ == "__main__":
    
    SITE, FOLDER = config()

    diff = SITE["end_date"] - SITE["start_date"]
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    print( hours,minutes,seconds)
