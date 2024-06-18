"""location specific settings used to initialise icestupa object
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

# Module logger = logging.getLogger("__main__")

def config(location=None, start_year=None, end_year=None, coords=None, alt=None, datadir=None):
    # logger.warning(f"Site {location}")

    if location == "leh":

        SITE = dict(
            name=location,
            alt=4009 if alt is None else alt,
            coords=[34.216638,77.606949] if coords is None else coords,
        )

    elif location == "europe":

        SITE = dict(
            name=location,
            alt=1013 if alt is None else alt,
            coords=[67.25, 17.75] if coords is None else coords,
        )

    elif location == "central_asia":

        SITE = dict(
            name=location,
            alt=998 if alt is None else alt,
            coords=[69.25, 95.25]  if coords is None else coords,
        )

    elif location == "south_america":

        SITE = dict(
            name=location,
            alt=3877 if alt is None else alt,
            coords=[-29.75, -69.75]  if coords is None else coords,
        )

    elif location == "north_america":

        SITE = dict(
            name=location,
            alt=1439 if alt is None else alt,
            coords=[-29.75, -69.75]  if coords is None else coords,
        )

    else:

        SITE = dict(
            name=location,
            alt=alt,
            coords=coords,
        )

    add = dict(
        # Calibrated values
        start_date=datetime(2019, 1, 1) if start_year is None else datetime(int(start_year),1,1),
        expiry_date =datetime(2020, 8, 31) if end_year is None else datetime(int(end_year)+1,1,1),
        R_F=10,
        V_dome=0,
        T_F=0,
        # cld=0.2,
        minimum_period=7,
    )

    SITE = dict(SITE, **add)
    # print(SITE)

    if datadir == None:
        FOLDER = dict(
            raw="data/" + location + "/raw/",
            input="data/" + location + "/interim/",
            output="data/" + location + "/processed/",
            fig="data/" + location + "/figs/",
        )
    else:
        FOLDER = dict(
            raw=datadir + "/era5/",
            input=datadir + location + "/interim/",
            output=datadir + location + "/processed/",
            fig=datadir +  location + "/figs/",
        )

    return SITE, FOLDER
