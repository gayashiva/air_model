"""Location specific settings used to initialise icestupa object
"""

# External modules
import pandas as pd
from datetime import datetime
import logging
import os, sys

# Module logger
logger = logging.getLogger(__name__)

# Spammers
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)


def config(location="Schwarzsee 2019", trigger="Manual"):

    logger.info("Location is %s and trigger is %s" % (location, trigger))

    if location == "Guttannen 2021" or location == "guttannen21":

        SITE = dict(
            name="guttannen21",
            start_date=datetime(2020, 11, 22),
            end_date=datetime(2021, 5, 18),
            fountain_off_date=datetime(2021, 2, 21),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT

            discharge=10,  # FOUNTAIN mean discharge
            min_discharge=5,  # FOUNTAIN min discharge
            # min_discharge=0,  # FOUNTAIN min discharge
            # perimeter=45, # on Feb 11
            # dia_f=0.00785,  # FOUNTAIN aperture diameter
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
            start_date=datetime(2020, 1, 3,16),
            end_date=datetime(2020, 4, 10),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT

            fountain_off_date=datetime(2020, 2, 27),
            min_discharge=5,  # FOUNTAIN min discharge
            # min_discharge=0,  # FOUNTAIN min discharge
            # perimeter=28, # on 24 Jan
            # dia_f=0.0056,  # FOUNTAIN aperture diameter
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 2.5},
            {"When": datetime(2020, 1, 24, 12), "h_f": 3.5},
            {"When": datetime(2020, 2, 5, 19), "h_f": 2.5},
        ]

    if location == "Schwarzsee 2019" or location == "schwarzsee19":
        SITE = dict(
            name="schwarzsee19",
            start_date=datetime(2019, 1, 30, 17),
            end_date=datetime(2019, 3, 17),
            # end_date=datetime(2019, 3, 20),
            utc_offset=1,
            longitude=7.297543,
            latitude=46.693723,
            h_aws=3,
            h_f=1.35,
            # T_W=1,

            fountain_off_date=datetime(2019, 2, 16, 10),
            discharge=3.58,  # FOUNTAIN on mean discharge from field
            dia_f=0.0056,  # FOUNTAIN aperture diameter
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 1.35},
        ]

    if location == "Gangles 2021" or location == "gangles21":

        SITE = dict(
            name="gangles21",
            # start_date=datetime(2020, 12, 14),
            start_date=datetime(2021, 1, 18),
            # end_date=datetime(2021, 3, 14),
            end_date=datetime(2021, 6, 11),
            utc_offset=5.5,
            longitude=77.606949,
            latitude=34.216638,
            h_aws=3,
            # T_W = 1,
            # dome_rad=2,

            fountain_off_date=datetime(2021, 3, 10, 18),
            discharge=60,  # FOUNTAIN on mean discharge from field
            # r_spray = 13.11, # On 3 Mar
            # perimeter=82.3, # On 3 Mar
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5},
            # {"When": datetime(2021, 1, 22, 16), "h_f": 9},
        ]
    if location == "Diavolezza 2021" or location == "diavolezza21":

        SITE = dict(
            name="diavolezza21",
            # start_date=datetime(2021, 1, 19,16),
            start_date=datetime(2021, 1, 26,9),
            # start_date=datetime(2021, 3, 1,16),
            # end_date=datetime(2021, 3, 18),
            # end_date=datetime(2021, 5, 18),
            end_date=datetime(2021, 5, 29),
            fountain_off_date=datetime(2021, 5, 1),
            utc_offset=1,
            latitude=46.44109,
            longitude=9.98425,
            h_aws=2.3, 

            # dome_rad=2,
            # perimeter=65.5, # on May 19
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5}, # doesnt use this?
        ]

    if location == "Ravat 2020" or location == "ravat20":

        SITE = dict(
            name="ravat20",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 8, 1),
            fountain_off_date=datetime(2020, 4, 1),
            utc_offset=6,
            latitude=39.87112,
            longitude=70.170666,
            h_aws=2, 

            discharge=10,  # FOUNTAIN mean discharge
            r_spray=7,  # FOUNTAIN mean discharge
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5}, # doesnt use this?
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
