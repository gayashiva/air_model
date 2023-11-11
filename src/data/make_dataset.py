"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os, json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.field import get_field
from src.data.era5 import get_era5
from src.data.meteoswiss import get_meteoswiss
from src.plots.data import plot_input
from src.utils import setup_logger

def e_sat(T, surface="water", a1=611.21, a3=17.502, a4=32.19):
    T += 273.16
    if surface == "ice":
        a1 = 611.21  # Pa
        a3 = 22.587  # NA
        a4 = -0.7  # K
    return a1 * np.exp(a3 * (T - 273.16) / (T - a4))

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    locations = ["south_america20", "north_america20", "europe20", "central_asia20", "leh20"]
    # locations = ["north_america20", "europe20", "central_asia20", "leh20"]
    # locations = ["leh20", "south_america20"]
    spray="ERA5_"

    with open("constants.json") as f:
        CONSTANTS = json.load(f)

    for loc in locations:
        print(loc)
        SITE, FOLDER = config(loc, spray)

        df= pd.read_csv(
            "data/era5/" + loc + ".csv",
            sep=",",
            header=0,
            parse_dates=["time"],
        )
        df = df.drop(['Unnamed: 0'], axis=1)

        df = df.set_index("time")

        time_steps = 60 * 60 
        df["ssrd"] /= time_steps
        df['wind'] = np.sqrt(df.u10**2 + df.v10**2)
        # Derive RH
        df["t2m"] -= 273.15
        df["d2m"] -= 273.15
        df["t2m_RH"] = df["t2m"].copy()
        df["d2m_RH"] = df["d2m"].copy()
        df= df.apply(lambda x: e_sat(x) if x.name == "t2m_RH" else x)
        df= df.apply(lambda x: e_sat(x) if x.name == "d2m_RH" else x)
        df["RH"] = 100 * df["d2m_RH"] / df["t2m_RH"]
        df = df.drop(['u10', 'v10', 't2m_RH', 'd2m_RH', 'd2m'], axis=1)
        df = df.reset_index()


        # CSV output
        df.rename(
            columns={
                "t2m": "temp",
                "ssrd": "SW_global",
            },
            inplace=True,
        )

        df = df.round(3)

        # logger.error(df.ssrd.mean())
        # logger.error(df.ssrd.max())
        df["Discharge"] = 1000000

        cols = [
            "time",
            "temp",
            "RH",
            "wind",
            "tcc",
            "SW_global",
            "Discharge",
        ]
        df_out = df[cols]

        if df_out.isna().values.any():
            logger.warning(df_out[cols].isna().sum())
            df_out = df_out.interpolate(method='ffill', axis=0)

        df_out = df_out.round(3)
        if len(df_out[df_out.index.duplicated()]):
            logger.error("Duplicate indexes")

        logger.error(df_out.SW_global.loc[2137])
        logger.error(df_out.SW_global.mean())
        logger.error(df_out.SW_global.max())
        logger.info(df_out.head(10))
        logger.info(df_out.tail())
        plot_input(df_out, FOLDER['fig'], SITE["name"])

        if not os.path.exists(dirname + "/data/" + loc):
            logger.warning("Creating folders")
            os.mkdir(dirname + "/data/" + loc)
            os.mkdir(dirname + "/" + FOLDER["input"])
            os.mkdir(dirname + "/" + FOLDER["output"])
            os.mkdir(dirname + "/" + FOLDER["fig"])

        df_out.to_csv(FOLDER["input"]  + "aws.csv", index=False)
