import os, sys
from datetime import datetime

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)
import logging

DIRNAME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SITE = "guttannen"

if SITE == "gangles":
    SITE = dict(
        name="gangles",
        end_date=datetime(2021, 2, 22),
        start_date=datetime(2021, 1, 30, 17),
        utc_offset=5.5,
        longitude=77.606949,
        latitude=34.216638,
        h_aws=3,
    )

    FOUNTAIN = dict(
        fountain_off_date=datetime(2021, 3, 10, 18),
        dia_f=0.005,  # FOUNTAIN aperture diameter
        h_f=1.35,  # FOUNTAIN steps h_f
        theta_f=45,  # FOUNTAIN aperture diameter
        ftl=0,  # FOUNTAIN flight time loss ftl
        T_w=5,  # FOUNTAIN Water temperature
        discharge=3.58,  # FOUNTAIN on discharge
        crit_temp=0,  # FOUNTAIN runtime temperature
    )

if SITE == "hial":
    SITE = dict(
        name="hial",
        end_date=datetime(2021, 2, 22),
        start_date=datetime(2021, 1, 30, 17),
        utc_offset=5.5,
        longitude=77.435594,
        latitude=34.160637,
        h_aws=3,
    )

    FOUNTAIN = dict(
        fountain_off_date=datetime(2021, 3, 10, 18),
        dia_f=0.005,  # FOUNTAIN aperture diameter
        h_f=1.35,  # FOUNTAIN steps h_f
        theta_f=45,  # FOUNTAIN aperture diameter
        ftl=0,  # FOUNTAIN flight time loss ftl
        T_w=5,  # FOUNTAIN Water temperature
        discharge=3.58,  # FOUNTAIN on discharge
        crit_temp=0,  # FOUNTAIN runtime temperature
    )

if SITE == "secmol":
    SITE = dict(
        name="secmol",
        end_date=datetime(2021, 2, 22),
        start_date=datetime(2021, 1, 30, 17),
        utc_offset=5.5,
        longitude=77.444852,
        latitude=34.130649,
        h_aws=3,
    )

    FOUNTAIN = dict(
        fountain_off_date=datetime(2021, 3, 10, 18),
        dia_f=0.005,  # FOUNTAIN aperture diameter
        h_f=1.35,  # FOUNTAIN steps h_f
        theta_f=45,  # FOUNTAIN aperture diameter
        ftl=0,  # FOUNTAIN flight time loss ftl
        T_w=5,  # FOUNTAIN Water temperature
        discharge=3.58,  # FOUNTAIN on discharge
        crit_temp=0,  # FOUNTAIN runtime temperature
    )

if SITE == "schwarzsee":
    SITE = dict(
        name="schwarzsee",
        end_date=datetime(2019, 3, 17),
        start_date=datetime(2019, 1, 30, 17),
        utc_offset=1,
        longitude=7.297543,
        latitude=46.693723,
        h_aws=3,
    )

    FOUNTAIN = dict(
        fountain_off_date=datetime(2019, 3, 10, 18),
        dia_f=0.005,  # FOUNTAIN aperture diameter
        h_f=1.35,  # FOUNTAIN steps h_f
        theta_f=45,  # FOUNTAIN aperture diameter
        ftl=0,  # FOUNTAIN flight time loss ftl
        T_w=5,  # FOUNTAIN Water temperature
        discharge=3.58,  # FOUNTAIN on discharge
        crit_temp=0,  # FOUNTAIN runtime temperature
    )

if SITE == "leh":
    SITE = dict(
        name="leh",
        end_date=datetime(2019, 4, 9),
        start_date=datetime(2019, 1, 30, 17),
        utc_offset=1,
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

if SITE == "schwarzsee_2020":
    FOLDERS = dict(
        DIRNAME=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        input_folder=os.path.join(DIRNAME, "data/interim/schwarzsee_2020/"),
        output_folder=os.path.join(DIRNAME, "data/processed/schwarzsee_2020/"),
        sim_folder=os.path.join(DIRNAME, "data/processed/schwarzsee_2020/simulations/"),
        data=os.path.join(DIRNAME, "data/raw/schwarzsee/CR6_DATA/CardConvert/"),
    )

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


if SITE == "guttannen":

    SITE = dict(
        name="guttannen",
        start_date=datetime(2020, 1, 1, 18),
        end_date=datetime(2020, 5, 1),
        error_date=datetime(2020, 1, 19),
        utc_offset=1,
        latitude=46.649999,
        longitude=8.283333,
        h_aws=3,
    )

    FOUNTAIN = dict(
        fountain_off_date=datetime(2019, 3, 1),
        dia_f=0.005,  # FOUNTAIN aperture diameter
        h_f=3.93,  # FOUNTAIN steps h_f
        theta_f=0,  # FOUNTAIN aperture diameter
        ftl=0,  # FOUNTAIN flight time loss ftl
        T_w=5,  # FOUNTAIN Water temperature
        discharge=3.58,  # FOUNTAIN on discharge
        crit_temp=0,  # FOUNTAIN runtime temperature
        tree_height=1.93,
        tree_radius=4.13 / 2,
    )

FOLDERS = dict(
    raw_folder=os.path.join(DIRNAME, "data/" + SITE["name"] + "/raw/"),
    input_folder=os.path.join(DIRNAME, "data/" + SITE["name"] + "/interim/"),
    output_folder=os.path.join(DIRNAME, "data/" + SITE["name"] + "/processed/"),
    sim_folder=os.path.join(DIRNAME, "data/" + SITE["name"] + "/processed/simulations"),
)
