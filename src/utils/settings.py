"""Location specific settings used to initialise icestupa object
"""

# External modules
import pandas as pd
from datetime import datetime
import logging
import os

# Module logger
logger = logging.getLogger(__name__)

# Spammers
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def config(location="Schwarzsee 2019", trigger="Manual"):

    logger.info("Location is %s and trigger is %s" % (location, trigger))
    if location == "Diavolezza 2021" or location == "diavolezza21":

        SITE = dict(
            name="diavolezza21",
            start_date=datetime(2021, 1, 19,16),
            # start_date=datetime(2021, 1, 26,9),
            # start_date=datetime(2021, 1, 19),
            end_date=datetime(2021, 3, 18),
            fountain_off_date=datetime(2021, 3, 18),
            utc_offset=1,
            latitude=46.44109,
            longitude=9.98425,
            h_aws=2.3, 

            # dia_f=0.008,  # FOUNTAIN aperture diameter
            dome_rad=2,
            meas_circum=65.5, # on May 19
            # DX = 14e-03  # Initial Ice layer thickness
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5}, # doesnt use this?
        ]


    if location == "Guttannen 2021":

        SITE = dict(
            name="guttannen21",
            start_date=datetime(2020, 11, 22),
            end_date=datetime(2021, 4, 26, 23),
            fountain_off_date=datetime(2021, 2, 21),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT
            # altitude_aws=1054,

            dia_f=0.00785,  # FOUNTAIN aperture diameter
            min_discharge=5,  # FOUNTAIN min discharge
            # meas_circum=45, # on Feb 11
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 2.5},
            {"When": datetime(2020, 12, 30, 16), "h_f": 3.5},
            {"When": datetime(2021, 1, 7, 16), "h_f": 5.5},
            {"When": datetime(2021, 1, 11, 16), "h_f": 4.5},
        ]

    if location == "Guttannen 2020":

        SITE = dict(
            name="guttannen20",
            start_date=datetime(2020, 1, 3,16),
            end_date=datetime(2020, 4, 6),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT
            # altitude_aws=1054,

            fountain_off_date=datetime(2020, 2, 27),
            dia_f=0.0056,  # FOUNTAIN aperture diameter
            min_discharge=5,  # FOUNTAIN min discharge
            # meas_circum=28, # on 24 Jan
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 2.5},
            {"When": datetime(2020, 1, 24, 12), "h_f": 3.5},
            {"When": datetime(2020, 2, 5, 19), "h_f": 2.5},
        ]

    if location == "Schwarzsee 2019":
        SITE = dict(
            name="schwarzsee19",
            start_date=datetime(2019, 1, 30, 17),
            end_date=datetime(2019, 3, 17),
            utc_offset=1,
            longitude=7.297543,
            latitude=46.693723,
            h_aws=3,

            fountain_off_date=datetime(2019, 3, 10, 18),
            discharge=3.58,  # FOUNTAIN on mean discharge from field
            dia_f=0.0056,  # FOUNTAIN aperture diameter
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 1.35},
        ]

    if location == "Gangles 2021":
        SITE = dict(
            name="gangles21",
            # start_date=datetime(2020, 12, 14),
            start_date=datetime(2021, 1, 18),
            end_date=datetime(2021, 3, 14),
            utc_offset=5.5,
            longitude=77.606949,
            latitude=34.216638,
            h_aws=3,
            dome_rad=2,

            fountain_off_date=datetime(2021, 3, 10, 18),
            meas_circum=82.3,
            dia_f=0.010,  # FOUNTAIN aperture diameter
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5},
            # {"When": datetime(2021, 1, 22, 16), "h_f": 9},
        ]

    # Define directory structure
    FOLDER = dict(
        raw="data/" + SITE["name"] + "/raw/",
        input="data/" + SITE["name"] + "/interim/",
        output="data/" + SITE["name"] + "/processed/",
        sim="data/" + SITE["name"] + "/processed/simulations/",
    )
    df_h = pd.DataFrame(data_h)

    return SITE, FOLDER, df_h
