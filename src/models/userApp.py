"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time, argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.utils.eff_criterion import nse
import logging, coloredlogs


def parse_args():
    parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

    parser.add_argument("--location", required=True, help="Specify the location filename as lat_long_alt (e.g., 34.216_77.606_4009.csv)")
    parser.add_argument("--start_year", required=False, help="Specify the start year (e.g., 2019)")
    parser.add_argument("--end_year", required=False, help="Specify the end year (e.g., 2020)")
    parser.add_argument("--datadir", required=False, help="Specify the data folder")

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



if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    # logger.setLevel("ERROR")
    logger.setLevel("WARNING")
    st = time.time()

    args = parse_args()
    coords, alt = extract_location_details(args.location)
    args.coords = coords
    args.alt = alt

    SITE, FOLDER = config(args.location, start_year=args.start_year, end_year=args.end_year, alt=args.alt,
                          coords=args.coords, datadir = args.datadir)
    icestupa = Icestupa(SITE, FOLDER)
    icestupa.sim_air(test=False)
    # icestupa.read_output()
    icestupa.summary_figures()
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = (et - st)/60
    print('\n\tExecution time:', round(elapsed_time,2), 'min\n')
