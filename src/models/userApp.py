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

# def parse_args():
#     parser = argparse.ArgumentParser(description="Command line interface to create or display Icestupa class")

#     parser.add_argument("--location", required=True, help="Specify the location (e.g., leh20)")

#     # Add more arguments as needed, such as start date, end date, altitude, etc.
#     parser.add_argument("--start_year", required=True, help="Specify the location (e.g., 2019)")
#     parser.add_argument("--end_year", required=True, help="Specify the location (e.g., 2020)")
#     parser.add_argument("--alt", required=False, help="Specify the location (e.g., 4000)")
#     parser.add_argument("--coords", required=False, help="Specify the location (e.g.,[43,108] )")

#     return parser.parse_args()

# def merge_with_settings(args):
#     # Merge command-line arguments with settings file
#     SITE, FOLDER = config(args.location)

#     # If the argument is not provided, use the value from the settings file
#     # You can extend this logic for other parameters
#     if not hasattr(args, 'start_year') or args.start_year is None:
#         args.start_date = SITE.get('start_year', None)
#         logger.warning(f"Default Start Year {args.start_date}")

#     if not hasattr(args, 'end_year') or args.end_year is None:
#         args.end_date = SITE.get('end_year', None)
#         logger.warning(f"Default End Year {args.end_date}")

#     if not hasattr(args, 'alt') or args.alt is None:
#         args.altitude = SITE.get('alt', None)
#         logger.warning(f"Default Altitude {args.altitude}")

#     if not hasattr(args, 'coords') or args.alt is None:
#         args.coords= SITE.get('coords', None)
#         logger.warning(f"Default coords {args.coords}")

#     return args, SITE, FOLDER

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
    # logger.setLevel("ERROR")
    logger.setLevel("WARNING")
    st = time.time()

    args = parse_args()

    # Merge command-line arguments with settings file
    args, SITE, FOLDER = merge_with_settings(args)

    # locations = ["north_america20"]
    # locations = ["leh20"]

    # locations = [ "north_america20", "europe20", "central_asia20","leh20", "south_america20"]

    # spray = "ERA5_"

    # for location in locations:

    # icestupa = Icestupa(args.location)
    SITE, FOLDER = config(args.location, start_year=args.start_year, end_year=args.end_year, alt=args.alt, coords=args.coords)
    icestupa = Icestupa(SITE, FOLDER)
    icestupa.sim_air(test=False)
    # icestupa.read_output()
    icestupa.summary_figures()
    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = (et - st)/60
    print('\n\tExecution time:', round(elapsed_time,2), 'min\n')
