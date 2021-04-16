import pandas as pd
import numpy as np
import logging
import coloredlogs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.data.settings import config

plt.rcParams["figure.figsize"] = (10, 7)
# mpl.rc('xtick', labelsize=5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)


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
    filename = FOLDER["sim"] + "/DX_sim.csv"

    filename2 = FOLDER["sim"] + "/DX_sim.h5"

    figures = FOLDER["sim"] + "/DX_sim.pdf"

    df = pd.read_csv(filename, sep=",")

    logger.info(df)
    pp = PdfPages(figures)
    x = df["DX"] * 1000
    y = df["Max_IceV"]

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_ylabel("Maximum Ice Volume ($m^3$)")
    ax.set_xlabel("Ice Layer Thickness ($mm$)")
    ax.grid()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    plt.close()
    pp.close()
