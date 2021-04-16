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
    # answers = dict(
    #     location="Schwarzsee 2019",
    #     location="Guttannen 2021",
    #     location="Gangles 2021",
    # trigger="Manual",
    # trigger="None",
    # trigger="Temperature",
    # trigger="Weather",
    # run="yes",
    # )
    locations = ["Schwarzsee 2019", "Guttannen 2021", "Guttannen 2020"]

    figures = "data/DX_sim.pdf"
    pp = PdfPages(figures)
    plt.figure()
    for location in locations:
        logger.info(f"Location -> %s" % (location))
        # Get settings for given location and trigger
        SITE, FOUNTAIN, FOLDER = config(location, "Manual")

        filename = FOLDER["sim"] + "/DX_sim.csv"

        df = pd.read_csv(filename, sep=",")

        logger.info(df)
        x = df["DX"] * 1000
        y = df["Max_IceV"]/ df.Max_IceV.max()

        plt.scatter(x, y, label=location)
        plt.text(x[0],y[0],str(round(df.loc[0,"Max_IceV"],2)))
        plt.text(x[38],y[38],str(round(df.loc[38,"Max_IceV"],2)))
        plt.ylabel("Maximum Ice Volume Sensitivity ($m^3$)")
        plt.xlabel("Ice Layer Thickness ($mm$)")
    plt.legend()
    plt.grid()
    pp.savefig(bbox_inches="tight")

    plt.close()
    pp.close()
