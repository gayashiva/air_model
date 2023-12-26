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

def config(loc=None, start_date=None, end_date=None, alt=None):
    # logger.warning(f"Site {loc} with scheduler {spray}")

    if loc == "leh20":

        SITE = dict(
            name=loc,
            alt=4009 if alt is None else alt,
            coords=[34.216638,77.606949] if coords is None else coords,
        )

    else if loc == "europe20":

        SITE = dict(
            name=loc,
            alt=1013 if alt is None else alt,
            coords=[67.25, 17.75] if coords is None else coords,
        )

    else if loc == "central_asia20":

        SITE = dict(
            name=loc,
            alt=998 if alt is None else alt,
            coords=[69.25, 95.25]  if coords is None else coords,
        )

    else if loc == "south_america20":

        SITE = dict(
            name="south_america20",
            alt=3877 if alt is None else alt,
            coords=[-29.75, -69.75]  if coords is None else coords,
        )

    else if loc == "north_america20":

        SITE = dict(
            name=loc,
            alt=1439 if alt is None else alt,
            coords=[-29.75, -69.75]  if coords is None else coords,
        )

    else:

        SITE = dict(
            name=loc,
            alt=alt,
            coords=coords,
        )

    add = dict(
        # Calibrated values
        start_date=datetime(2021, 1, 1) if start_date is None else start_date,
        expiry_date =datetime(2022, 12, 31) if expiry_date is None else expiry_date,
        R_F=10,
        V_dome=0,
        T_F=0,
        # cld=0.2,
        minimum_period=7 if minimum_period is None else minimum_period,
    )

    SITE = dict(SITE, **add)

    FOLDER = dict(
        raw="data/" + SITE["name"] + "/raw/",
        input="data/" + SITE["name"] + "/interim/",
        output="data/" + SITE["name"] + "/processed/",
        fig="data/" + SITE["name"] + "/figs/",
    )

    if not os.path.exists(dirname + "data/" + SITE["name"]):
        os.mkdir(dirname + FOLDER["raw"])
        os.mkdir(dirname + FOLDER["input"])
        os.mkdir(dirname + FOLDER["output"])
        os.mkdir(dirname + FOLDER["fig"])


    return SITE, FOLDER
