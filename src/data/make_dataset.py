"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
import math
import time
from pathlib import Path
from tqdm import tqdm
import logging
import coloredlogs
from scipy import stats
from sklearn.linear_model import LinearRegression

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config
from src.data.field import get_field
from src.data.era5 import get_era5
from src.data.meteoswiss import get_meteoswiss
from src.data.discharge import get_discharge

def linreg(X, Y):
    mask = ~np.isnan(X) & ~np.isnan(Y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[mask], Y[mask])
    return slope, intercept

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(name)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )
    # location = "guttannen21"
    # location = "schwarzsee19"
    location = "gangles21"


    SITE, FOLDER = config(location)

    if location in ["gangles21"]:
        df = get_field(location)
        df = df.set_index("When")
        df = df[SITE['start_date']:SITE["end_date"]]
        df = df.reset_index()
        logger.info(df.missing_type.describe())
        logger.info(df.missing_type.unique())
    else:

        if location in ["schwarzsee19", "gangles21"]:
            df = get_field(location)

        if location in ["guttannen21", "guttannen20"]:
            df = get_meteoswiss(location)

        df = df.set_index("When")
        df = df[SITE['start_date']:SITE["end_date"]]
        df = df.reset_index()

        # Replace Wind zero values for 3 hours
        mask = df.v_a.shift().eq(df.v_a)
        for i in range(1,3*4):
            mask &= df.v_a.shift(-1 * i).eq(df.v_a)
        mask &= (df.v_a ==0)
        df.v_a = df.v_a.mask(mask)

        if location in ["schwarzsee19"]:
            df_swiss = get_meteoswiss(location)
            df_swiss = df_swiss.set_index("When")
            df_swiss = df_swiss[SITE['start_date']:SITE["end_date"]]
            df_swiss = df_swiss.reset_index()
            print(df_swiss.shape[0])

            df_swiss = df_swiss.set_index("When")
            df= df.set_index("When")

            for col in ["Prec"]:
                logger.info("%s from meteoswiss" % col)
                df[col] = df_swiss[col]
            df_swiss = df_swiss.reset_index()
            df= df.reset_index()


        df_ERA5_full = get_era5(SITE["name"])

        df = df.set_index("When")
        df_ERA5_full = df_ERA5_full.set_index("When")
        df_ERA5 = df_ERA5_full[SITE['start_date']:SITE["end_date"]]
        df_ERA5 = df_ERA5.reset_index()
        df_ERA5_full = df_ERA5_full.reset_index()
        print(df_ERA5.shape[0])

        # Fit ERA5 to field data
        if SITE["name"] in ["guttannen21", "guttannen20"]:
            fit_list = ["T_a", "RH", "v_a"]

        if SITE["name"] in ["schwarzsee19"]:
            fit_list = ["T_a", "RH", "v_a", "p_a"]

        for column in fit_list:
            Y = df[column].values.reshape(-1, 1)
            X = df_ERA5[column].values.reshape(-1, 1)
            slope, intercept = linreg(X, Y)
            df_ERA5[column] = slope * df_ERA5[column] + intercept
            df_ERA5_full[column] = slope * df_ERA5_full[column] + intercept
            if column in ["v_a"]:
                # Correct negative wind
                df_ERA5.v_a.loc[df_ERA5.v_a<0] = 0
                df_ERA5_full.v_a.loc[df_ERA5_full.v_a<0] = 0

        df_ERA5 = df_ERA5.set_index("When")

        # Fill from ERA5
        df['missing_type'] = ''
        for col in ["T_a", "RH", "v_a", "Prec", "p_a", "SW_direct", "SW_diffuse", "LW_in"]:
            try:
                mask = df[col].isna()
                percent_nan = df[col].isna().sum()/df.shape[0] * 100
                logger.info(" %s has %s percent NaN values" %(col, percent_nan))
                if percent_nan > 1 :
                    logger.warning(" Null values filled with ERA5 in %s" %col)
                    df.loc[df[col].isna(), "missing_type"] = df.loc[df[col].isna(), "missing_type"] + col
                    df.loc[df[col].isna(), col] = df_ERA5[col]
                else:
                    logger.warning(" Null values interpolated in %s" %col)
                    df.loc[:, col] = df[col].interpolate()
            except KeyError:
                logger.warning("%s from ERA5" % col)
                df[col] = df_ERA5[col]
                df["missing_type"] = df["missing_type"] + col
        logger.info(df.missing_type.describe())
        logger.info(df.missing_type.unique())

        df = df.reset_index()

    if SITE["name"] in ["gangles21"]:
        cols = [
            "When",
            # "Discharge",
            "T_a",
            "RH",
            "v_a",
            # "SW_direct",
            # "SW_diffuse",
            "SW_global",
            "Prec",
            # "vp_a",
            "p_a",
            "missing_type",
            # "LW_in",
            "cld",
        ]

    if SITE["name"] in ["schwarzsee19"]:
        cols = [
            "When",
            # "Discharge",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            # "vp_a",
            "p_a",
            "missing_type",
            "LW_in",
        ]
    if SITE["name"] in ["guttannen20", "guttannen21"]:
        cols = [
            "When",
            # "Discharge",
            "T_a",
            "RH",
            "v_a",
            "SW_direct",
            "SW_diffuse",
            "Prec",
            "vp_a",
            "p_a",
            "missing_type",
            "LW_in",
        ]

    df_out = df[cols]

    if df_out.isna().values.any():
        print(df_out[cols].isna().sum())
        for column in cols:
            if df_out[column].isna().sum() > 0 and column not in ["missing_type"]:
                logger.warning(" Null values interpolated in %s" %column)
                df_out.loc[:, column] = df_out[column].interpolate()

    df_out = df_out.round(3)
    if len(df_out[df_out.index.duplicated()]):
        logger.error("Duplicate indexes")


    logger.info(df_out.tail())
    df_out.to_csv(FOLDER["input"] + SITE["name"] + "_input_model.csv", index=False)

    fig = plt.figure()
    plt.plot(df_out.p_a)
    plt.ylabel('some numbers')
    plt.savefig(FOLDER["input"] + SITE["name"] + "test.png")

    # Extend field data with ERA5
    if SITE["name"] in ['schwarzsee19']:
        df_ERA5_full["Prec"] = 0
        df_ERA5_full["Discharge"] = 0
        df_ERA5_full["missing_type"] = "-".join(df_out.columns)
        mask = (df_ERA5_full["When"] > df_out["When"].iloc[-1]) & (
            df_ERA5_full["When"] <= datetime(2019, 4, 30)
        )
        df_ERA5_full = df_ERA5_full.loc[mask]

        df_out = df_out.set_index("When")
        df_ERA5_full = df_ERA5_full.set_index("When")

        df_swiss = get_meteoswiss(SITE["name"])
        df_swiss = df_swiss.set_index("When")
        df_swiss = df_swiss[SITE['start_date']:datetime(2019, 4, 30)]
        df_swiss = df_swiss.reset_index()

        df_swiss = df_swiss.set_index("When")

        df_ERA5_full["Prec"] = df_swiss["Prec"]
        concat = pd.concat([df_out, df_ERA5_full])
        if len(concat[concat.index.duplicated()]):
            logger.error("Duplicate indexes")
        logger.info(concat.tail())

        concat = concat.reset_index()

        if concat.isna().values.any():
            print(concat[cols].isna().sum())
            for column in cols:
                if concat[column].isna().sum() > 0 and column not in ["missing_type"]:
                    logger.warning(" Null values interpolated in %s" %column)
                    concat.loc[:, column] = concat[column].interpolate()

        print(concat.columns)
        concat.to_csv(FOLDER["input"] + SITE["name"] + "_input_model.csv", index=False)
        concat.to_hdf(
            FOLDER["input"] + SITE["name"] + "_input_model.h5",
            key="df",
            mode="w",
        )

