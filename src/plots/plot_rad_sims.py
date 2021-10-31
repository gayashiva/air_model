"""Create Slides
"""

# External modules
import os, sys
import matplotlib.pyplot as plt
import pandas as pd

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger
import logging, coloredlogs


if __name__ == "__main__":

    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("WARNING")

    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"

    SITE, FOLDER = config()

    df = pd.read_csv(FOLDER["sim"] + "rad_sims.csv")

    for ctr in range(2):
        x = df.rad

        y1 = df.maxV
        y2 = df.days
        fig, ax = plt.subplots()
        ax.axvline(
            x=7,
            color="grey",
            linestyle="--",
            zorder=0,
            label="Guttannen Spray radius",
        )
        ax.scatter(
            7,
            df.loc[df.rad == 7].maxV,
            s=100,
            color=CB91_Blue,
            marker="*",
            zorder=10,
        )
        ax.axvline(
            x=43,
            color="grey",
            linestyle="--",
            zorder=0,
            # label="Guttannen Spray radius",
        )
        ax.scatter(
            43,
            df.loc[df.rad == 43].maxV,
            s=100,
            color=CB91_Blue,
            marker="*",
            zorder=10,
        )
        ax.scatter(
            x,
            y1,
            s=5,
            color=CB91_Blue,
            alpha=ctr,
        )
        for tl in ax.get_yticklabels():
            tl.set_color(CB91_Blue)
        ax.set_ylabel("Max AIR Volume [$m^{3}$]", color=CB91_Blue)
        ax.set_xlabel("Fountain Spray Radius [$m$]")
        axt = ax.twinx()
        axt.scatter(
            7,
            df.loc[df.rad == 7].days,
            s=100,
            color=CB91_Purple,
            marker="*",
        )
        axt.scatter(
            43,
            df.loc[df.rad == 43].days,
            s=100,
            color=CB91_Purple,
            marker="*",
        )
        axt.scatter(
            x,
            y2,
            s=5,
            color=CB91_Purple,
            alpha=ctr,
        )
        axt.set_ylabel("Survival duration [$days$]", color=CB91_Purple)
        for tl in axt.get_yticklabels():
            tl.set_color(CB91_Purple)
        plt.savefig(FOLDER["sim"] + "rad_sims_" + str(ctr) + ".jpg")
