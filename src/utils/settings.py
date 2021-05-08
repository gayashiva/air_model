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
    if location == "Guttannen 2021":

        SITE = dict(
            name="guttannen21",
            start_date=datetime(2020, 11, 22),
            end_date=datetime(2021, 4, 26, 23),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT
            altitude_aws=1054,
            # DX = 10e-03,  # Initial Ice layer thickness
            # TIME_STEP = 15*60,  # Initial Ice layer thickness
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2021, 2, 21),
            dia_f=0.007,  # FOUNTAIN aperture diameter
            h_f=2.5,  # FOUNTAIN steps h_f
            discharge=12.33,  # FOUNTAIN on discharge
            trigger=trigger,
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 2.5},
            {"When": datetime(2020, 12, 30, 16), "h_f": 3.5},
            {"When": datetime(2021, 1, 11, 16), "h_f": 4.5},
            {"When": datetime(2021, 1, 7, 16), "h_f": 5.5},
        ]

    if location == "Guttannen 2020":

        SITE = dict(
            name="guttannen20",
            # start_date=datetime(2019, 12, 28),
            start_date=datetime(2020, 1, 3,16),
            end_date=datetime(2020, 4, 6),
            # end_date=datetime(2020, 1, 20),
            # end_date=datetime(2020, 2, 10),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT
            altitude_aws=1054,
            # DX = 10e-03,  # Initial Ice layer thickness
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2020, 2, 27),
            # fountain_off_date=datetime(2020, 3, 17, 18),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=2.5,  # FOUNTAIN steps h_f
            discharge=10,  # FOUNTAIN on discharge
            trigger=trigger,
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
            # DX = 4e-03,  # Initial Ice layer thickness
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2019, 3, 10, 18),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            # h_f=1.35,  # FOUNTAIN steps h_f
            discharge=3.58,  # FOUNTAIN on discharge
            trigger=trigger,
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
            dome_rad=4,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2021, 3, 10, 18),
            dia_f=0.01,  # FOUNTAIN aperture diameter
            h_f=5,  # FOUNTAIN steps h_f
            discharge=120,  # FOUNTAIN on discharge
            trigger=trigger,
        )

        data_h = [
            {"When": SITE["start_date"], "h_f": 5},
        ]

    # Define directory structure
    FOLDER = dict(
        raw="data/" + SITE["name"] + "/raw/",
        input="data/" + SITE["name"] + "/interim/",
        output="data/" + SITE["name"] + "/processed/",
        sim="data/" + SITE["name"] + "/processed/simulations",
    )
    df_h = pd.DataFrame(data_h)

    return SITE, FOUNTAIN, FOLDER, df_h
