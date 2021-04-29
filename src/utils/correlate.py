import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
import os, sys, logging, coloredlogs
from sklearn.metrics import mean_squared_error

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )

    answers = dict(
        # location="Schwarzsee 2019",
        location="Guttannen 2021",
        # location="Gangles 2021",
        trigger="Manual",
        # trigger="None",
        # trigger="Temperature",
        # trigger="Weather",
        run="yes",
    )

    # Get settings for given location and trigger
    SITE, FOUNTAIN, FOLDER = config(answers["location"], answers["trigger"])

    # Initialise icestupa object
    icestupa = Icestupa(answers["location"], answers["trigger"])
    icestupa.read_output()
    # icestupa.df = icestupa.df.set_index("When").resample("H").mean().reset_index()
    print(icestupa.df['DroneV'].corr(icestupa.df['iceV']))

    print(icestupa.df[['When', 'cam_temp']].loc[icestupa.df.cam_temp>0])
    pp = PdfPages(FOLDER["output"] + "correlations" + ".pdf")
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.scatter(icestupa.df.T_s, icestupa.df.cam_temp, s=2)
    ax1.set_xlabel("Modelled Temp")
    ax1.set_ylabel("Measured Temp")
    ax1.grid()

    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax1.plot(lims, lims, "--k", alpha=0.25, zorder=0)
    ax1.set_aspect("equal")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    # format the ticks

    pp.savefig(bbox_inches="tight")
    plt.clf()
    pp.close()
