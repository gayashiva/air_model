"""Command line interface to create or display Icestupa class
"""

# External modules
import os, sys, shutil, time
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
from src.plots.data import plot_input


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    # logger.setLevel("ERROR")
    logger.setLevel("WARNING")
    st = time.time()

    test = True
    # test = False

    # location = "Guttannen 2020"
    # location = "Guttannen 2021"
    # location = "Guttannen 2022"
    # location = "Gangles 2021"
    # location = "sibinacocha21"
    # location = "sibinacocha22"
    # location = "altiplano20"
    # location = "north_america20"
    # locations = ["south_america20"]
    # locations = ["leh20"]

    # locations = ["central_asia20"]
    # locations = ["chuapalca20"]
    # locations = ["candarave20"]
    locations = [ "north_america20", "europe20", "central_asia20","leh20", "south_america20"]

    # sprays = ["scheduled_icv", "scheduled_wue"]
    # sprays = ["unscheduled_field", "scheduled_field"]
    # sprays = ["unscheduled_field"]
    # sprays = ["scheduled_wue", "scheduled_icv"]
    # sprays = ["none_none"]
    sprays = ["ERA5_"]

    for location in locations:

        for spray in sprays:
            icestupa = Icestupa(location, spray)
            SITE, FOLDER = config(location)

            if test:
                # icestupa.gen_input()
                icestupa.read_input()
                # plot_input(icestupa.df, FOLDER['fig'], SITE["name"])
                # icestupa.sim_air(test)
                icestupa.sim_air()
                icestupa.gen_output()
                # icestupa.read_output()
                icestupa.summary_figures()
                # print(icestupa.df.s_cone.max())
                # get the end time
                et = time.time()

                # get the execution time
                elapsed_time = (et - st)/60
                print('Execution time:', round(elapsed_time), 'min')

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

