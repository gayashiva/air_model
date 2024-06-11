"""Compile raw data from the location, meteoswiss or ERA5
"""

# External modules
import sys, os, json, argparse
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

# def parse_args():
#     parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

#     parser.add_argument("--location", required=True, help="Specify the location as lat_long_alt.csv")

#     # Add more arguments as needed, such as start date, end date, altitude, etc.
#     parser.add_argument("--start_year", required=True, help="Specify the location (e.g., leh20)")
#     parser.add_argument("--end_year", required=True, help="Specify the location (e.g., leh20)")
#     parser.add_argument("--alt", required=True, help="Specify the location (e.g., leh20)")
#     parser.add_argument("--coords", required=True, help="Specify the location (e.g., leh20)")

#     return parser.parse_args()
def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

    parser.add_argument("--location", required=True, help="Specify the location filename as lat_long_alt (e.g., 34.216_77.606_4009.csv)")
    parser.add_argument("--start_year", required=True, help="Specify the start year (e.g., 2019)")
    parser.add_argument("--end_year", required=True, help="Specify the end year (e.g., 2020)")

    return parser.parse_args()

def extract_location_details(location):
    # Strip the file extension if present
    if location.endswith('.csv'):
        location = location

    # Split the string based on underscore
    try:
        lat, long, alt = location.split('_')
        coords = [float(lat), float(long)]
        alt = int(alt)
    except ValueError:
        raise ValueError("Location should be in the format lat_long_alt.csv (e.g., 34.216_77.606_4009.csv)")

    return coords, alt

def merge_with_settings(args):
    # Extract coordinates and altitude from the location filename
    coords, alt = extract_location_details(args.location)
    args.coords = coords
    args.alt = alt

    # Merge command-line arguments with settings file
    SITE, FOLDER = config(args.location)

    # If the argument is not provided, use the value from the settings file
    # You can extend this logic for other parameters
    if not hasattr(args, 'start_year') or args.start_year is None:
        args.start_date = SITE.get('start_year', None)

    if not hasattr(args, 'end_year') or args.end_year is None:
        args.end_date = SITE.get('end_year', None)

    # if not hasattr(args, 'alt') or args.alt is None:
    #     args.altitude = SITE.get('alt', None)

    # if not hasattr(args, 'coords') or args.coords is None:
    #     args.coords = SITE.get('coords', None)

    return args, SITE, FOLDER

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("INFO")

    args = parse_args()

    # Merge command-line arguments with settings file
    args, SITE, FOLDER = merge_with_settings(args)

    # locations = ["south_america20", "north_america20", "europe20", "central_asia20", "leh20"]

    with open("constants.json") as f:
        CONSTANTS = json.load(f)

    # for loc in locations:
    loc= args.location
    # print(*args)
    SITE, FOLDER = config(args.location, start_year=args.start_year, end_year=args.end_year, alt=args.alt, coords=args.coords)

    df= pd.read_csv(
        "data/era5/" + loc + ".csv",
        sep=",",
        header=0,
        parse_dates=["time"],
    )
    # df = df.drop(['Unnamed: 0'], axis=1)

    df = df.set_index("time")
    df = df[SITE["start_date"]:SITE["expiry_date"]]

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
        # "tcc",
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

    logger.info(df_out.head())
    logger.info(df_out.tail())

    if not os.path.exists(dirname + "/data/" + loc):
        logger.warning("Creating folders")
        os.mkdir(dirname + "/data/" + loc)
        os.mkdir(dirname + "/" + FOLDER["input"])
        os.mkdir(dirname + "/" + FOLDER["output"])
        os.mkdir(dirname + "/" + FOLDER["fig"])

    df_out.to_csv(FOLDER["input"]  + "aws.csv", index=False)
    plot_input(df_out, FOLDER['fig'], SITE["name"])
