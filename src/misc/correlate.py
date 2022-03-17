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
    logger.setLevel("WARNING")

    # location="Schwarzsee 2019"
    # location="phortse20"
    # location = "Guttannen 2020"
    # location = "Guttannen 2021"
    location = "Guttannen 2022"
    # location = "Gangles 2021"

    # Initialise icestupa object
    icestupa = Icestupa(location)
    icestupa.read_output()
    CONSTANTS, SITE, FOLDER = config(location)

    # print(icestupa.df['Qs_meas'].corr(icestupa.df['Qs']))
    # print(icestupa.df['cam_temp'].corr(icestupa.df['T_s']))
    # print(((icestupa.df.DroneV - icestupa.df.iceV) ** 2).mean() ** .5)

    # print(icestupa.df[['When', 'cam_temp']].loc[icestupa.df.cam_temp>0])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(icestupa.df.Qs, -icestupa.df.Qs_meas, s=2)
    ax1.set_xlabel("Modelled Qs")
    ax1.set_ylabel("Measured Qs")
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

    plt.savefig(
        FOLDER['fig'] + "correlate_Qs.jpg",
        bbox_inches="tight",
        dpi=300,
    )
    plt.clf()
