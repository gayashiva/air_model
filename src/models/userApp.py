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

    # sprays = ["manual", "dynamic"]
    # sprays = ["manual", "dynamic", "dynamic_field"]
    # sprays = ["dynamic_field"]
    # sprays=["manual"]
    sprays=["dynamic_field", "manual"]

    for spray in sprays:
        icestupa = Icestupa(location, spray)

        if test:
            icestupa.gen_input()
            icestupa.sim_air(test)
            icestupa.gen_output()
            # icestupa.read_output()
            icestupa.summary_figures()

            rmse = mean_squared_error(icestupa.df.T_bulk_meas, icestupa.df.T_bulk, squared=False)
            nse = nse(icestupa.df.T_bulk, icestupa.df.T_bulk_meas)
            print(f"Calculated NSE {nse} and RMSE {rmse}")


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
                except OSError as e:
                    # If the error was caused because the source wasn't a directory
                    if e.errno == errno.ENOTDIR:
                       shutil.copy(source_dir_prompt, destination_dir_prompt)
                    else:
                        print('Directory not copied. Error: %s' % e)

            icestupa.read_output()
            # df = icestupa.df
            # print(df.columns)
            # icestupa.summary_figures()



            # fig, ax = plt.subplots()
            # x1 = df.time
            # y1 = df.Discharge
            # # x2 = dfr.time
            # # y2 = dfr.rad
            # ax.plot(x1,y1)
            # # ax.scatter(x2,y2)

            # ax.xaxis.set_major_locator(mdates.WeekdayLocator())
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            # ax.xaxis.set_minor_locator(mdates.DayLocator())
            # fig.autofmt_xdate()
            # plt.savefig(
            #     icestupa.fig + icestupa.spray + "/discharge.jpg",
            #     bbox_inches="tight",
            # )
            # plt.clf()

