"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from sklearn.metrics import mean_squared_error


# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
from src.utils.eff_criterion import nse
import logging, coloredlogs


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")
    # logger.setLevel("INFO")

    test = True
    # test = False

    # location = "Guttannen 2020"
    # location = "Guttannen 2021"
    location = "Guttannen 2022"
    # location = "Gangles 2021"

    # sprays = ["scheduled_icv", "scheduled_wue"]
    # sprays = ["scheduled_field", "unscheduled_field"]
    sprays = ["unscheduled_field"]
    # sprays = ["scheduled_wue", "scheduled_icv"]

    for spray in sprays:
        icestupa = Icestupa(location, spray)

        if test:
            icestupa.gen_input()
            icestupa.sim_air(test)
            icestupa.gen_output()
            # icestupa.read_output()
            icestupa.summary_figures()
            # print(icestupa.df.s_cone.max())

            # if location == "Guttannen 2022" and spray == "scheduled_field":
            #     rmse = mean_squared_error(icestupa.df.T_bulk_meas, (icestupa.df.T_bulk + icestupa.df.T_s)/2, squared=False)
            #     nse = nse(icestupa.df.T_bulk, (icestupa.df.T_bulk + icestupa.df.T_s)/2)
            #     print(f"Calculated NSE {nse} and RMSE {rmse}")

        else:
            # For web app
            src = "/home/suryab/work/air_model/data/" + icestupa.name + "/"
            dst = "/home/suryab/work/air_app/data/" + icestupa.name + "/"
            for dir in ["processed", "figs"]:
                try:
                    #if path already exists, remove it before copying with copytree()
                    if os.path.exists(dst + dir):
                        shutil.rmtree(dst + dir)
                        shutil.copytree(src + dir, dst + dir)
                    else:
                        shutil.copytree(src + dir, dst + dir)
                    print('Directory copied.')
                except OSError as e:
                    # If the error was caused because the source wasn't a directory
                    if e.errno == errno.ENOTDIR:
                       shutil.copy(source_dir_prompt, destination_dir_prompt)
                    else:
                        print('Directory not copied. Error: %s' % e)

            icestupa.read_output()

