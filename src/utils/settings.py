"""Location specific settings used to initialise icestupa object
"""

# External modules
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
            end_date=datetime(2021, 3, 19),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT
            altitude_aws=1054,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2021, 2, 21),
            dia_f=0.007,  # FOUNTAIN aperture diameter
            h_f=2.5,  # FOUNTAIN steps h_f
            discharge=12.33,  # FOUNTAIN on discharge
            trigger=trigger,
        )

    if location == "Guttannen 2020":

        SITE = dict(
            name="guttannen20",
            # start_date=datetime(2019, 12, 28),
            start_date=datetime(2020, 1, 3,16),
            end_date=datetime(2020, 4, 6),
            # end_date=datetime(2020, 2, 10),
            utc_offset=1,
            latitude=46.649999,
            longitude=8.283333,
            h_aws=2.3,  # https://www.meteoschweiz.admin.ch/home/messwerte.html?param=messnetz-partner&chart=day&table=true&sortDirection=&station=MMGTT
            altitude_aws=1054,
            # hollowV=44.6,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2020, 2, 27),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=2.5,  # FOUNTAIN steps h_f
            discharge=10,  # FOUNTAIN on discharge
            trigger=trigger,
        )

    if location == "Schwarzsee 2019":
        SITE = dict(
            name="schwarzsee19",
            start_date=datetime(2019, 1, 30, 17),
            end_date=datetime(2019, 3, 17),
            utc_offset=1,
            longitude=7.297543,
            latitude=46.693723,
            h_aws=3,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2019, 3, 10, 18),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=1.35,  # FOUNTAIN steps h_f
            discharge=3.58,  # FOUNTAIN on discharge
            trigger=trigger,
        )

    if location == "Gangles 2021":
        SITE = dict(
            name="gangles21",
            end_date=datetime(2021, 3, 14),
            # start_date=datetime(2020, 12, 14),
            start_date=datetime(2021, 1, 18),
            utc_offset=5.5,
            longitude=77.606949,
            latitude=34.216638,
            h_aws=3,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2021, 3, 10, 18),
            dia_f=0.01,  # FOUNTAIN aperture diameter
            h_f=5,  # FOUNTAIN steps h_f
            discharge=60,  # FOUNTAIN on discharge
            trigger=trigger,
        )
    if location == "Hial":

        SITE = dict(
            name="hial",
            # end_date=datetime(2021, 2, 22),
            start_date=datetime(2021, 1, 30, 17),
            utc_offset=5.5,
            longitude=7.297543,
            latitude=46.693723,
            h_aws=3,
        )

        FOUNTAIN = dict(
            # fountain_off_date=datetime(2021, 3, 10, 18),
            dia_f=0.01,  # FOUNTAIN aperture diameter
            h_f=2,  # FOUNTAIN steps h_f
            theta_f=45,  # FOUNTAIN aperture diameter
            ftl=0.5,  # FOUNTAIN flight time loss ftl
            T_w=5,  # FOUNTAIN Water temperature
            discharge=60,  # FOUNTAIN on discharge
            crit_temp=-1,  # FOUNTAIN runtime temperature
            trigger="NetEnergy",
        )
    if location == "Secmol":
        SITE = dict(
            name="secmol",
            # end_date=datetime(2021, 2, 22),
            start_date=datetime(2021, 1, 30, 17),
            utc_offset=5.5,
            longitude=77.444852,
            latitude=34.130649,
            h_aws=3,
        )

        FOUNTAIN = dict(
            # fountain_off_date=datetime(2021, 3, 10, 18),
            dia_f=0.01,  # FOUNTAIN aperture diameter
            h_f=2,  # FOUNTAIN steps h_f
            theta_f=45,  # FOUNTAIN aperture diameter
            ftl=0.5,  # FOUNTAIN flight time loss ftl
            T_w=5,  # FOUNTAIN Water temperature
            discharge=60,  # FOUNTAIN on discharge
            crit_temp=-1,  # FOUNTAIN runtime temperature
            trigger="NetEnergy",
        )

    if location == "Leh":
        SITE = dict(
            name="leh",
            end_date=datetime(2019, 4, 9),
            start_date=datetime(2019, 1, 30, 17),
            utc_offset=5.5,
            longitude=77.5771,
            latitude=34.1526,
            h_aws=3,
        )

        FOUNTAIN = dict(
            fountain_off_date=datetime(2019, 2, 16, 10),
            dia_f=0.005,  # FOUNTAIN aperture diameter
            h_f=1.35,  # FOUNTAIN steps h_f
            theta_f=45,  # FOUNTAIN aperture diameter
            ftl=0,  # FOUNTAIN flight time loss ftl
            T_w=5,  # FOUNTAIN Water temperature
            discharge=3.58,  # FOUNTAIN on discharge
            crit_temp=0,  # FOUNTAIN runtime temperature
        )

    if location == "Schwarzsee_2020":
        dates = dict(
            start_date=datetime(2020, 2, 15),
            end_date=datetime(2020, 2, 18),
            fountain_off_date=datetime(2020, 2, 10),
        )
        FOUNTAIN = dict(
            aperture_f=0.005,  # FOUNTAIN aperture diameter
            h_f=4,  # FOUNTAIN steps h_f
            discharge=3.58,  # FOUNTAIN on discharge
            crit_temp=-5,  # FOUNTAIN runtime temperature
            latitude=46.693723,
            longitude=7.297543,
            utc_offset=1,
        )

    # Define directory structure
    FOLDER = dict(
        # raw=os.path.join(dirname, "data/" + SITE["name"] + "/raw/"),
        # input=os.path.join(dirname, "data/" + SITE["name"] + "/interim/"),
        # output=os.path.join(dirname, "data/" + SITE["name"] + "/processed/"),
        # sim=os.path.join(dirname, "data/" + SITE["name"] + "/processed/simulations"),
        raw="data/" + SITE["name"] + "/raw/",
        input="data/" + SITE["name"] + "/interim/",
        output="data/" + SITE["name"] + "/processed/",
        sim="data/" + SITE["name"] + "/processed/simulations",
    )

    return SITE, FOUNTAIN, FOLDER
