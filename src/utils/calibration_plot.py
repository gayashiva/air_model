import pandas as pd
import logging
import coloredlogs
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.utils import setup_logger
from src.models.icestupaClass import Icestupa

plt.rcParams["figure.figsize"] = (10, 7)
# mpl.rc('xtick', labelsize=5)
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.ERROR,
        logger=logger,
    )
    answers = dict(
        # location="Schwarzsee 2019",
        location="Guttannen 2021",
        # location="Gangles 2021",
        trigger="Manual",
    )

    # Get settings for given location and trigger
    SITE, FOUNTAIN, FOLDER = config(answers["location"])

    # Initialise icestupa object
    icestupa = Icestupa(answers["location"])

    cmap = plt.cm.rainbow  # define the colormap
    norm = mpl.colors.Normalize(vmin=-100, vmax=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    filename = FOLDER["sim"] + "/DX_sim.csv"
    filename2 = FOLDER["sim"] + "/DX_sim.h5"
    figures = FOLDER["sim"] + "/DX_sim.pdf"
    filename = FOLDER["sim"] + "/Tune_sim.csv"
    figures = FOLDER["sim"] + "/Tune_sim.pdf"

    df = pd.read_csv(filename, sep=",")
    logger.info(df)

    # data_store = pd.HDFStore(filename2)
    # dfd = data_store["dfd"]
    # data_store.close()
    # print(dfd.head())

    x = df["DX"]
    x2 = df["TIME_STEP"]
    y = df["rmse_V"]
    # y2 = df["Min_T_bulk"]

    pp = PdfPages(figures)
    fig, ax = plt.subplots()
    for i in range(0, df.shape[0]):
        if x2[i] == 15:
            l1 = ax.scatter(x[i],y[i], facecolors="blue")
        if x2[i] == 30:
            l2 = ax.scatter(x[i],y[i], facecolors="red")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("Ice Layer Thickness ($mm$)")
    ax.legend(
        (l1, l2),
        ("15","30"),
        scatterpoints=1,
        loc="lower right",
        ncol=1,
        # fontsize=8
    )
    ax.grid()
    pp.savefig(bbox_inches="tight")
    plt.close()
    pp.close()

    # for location in locations:
    #     logger.info(f"Location -> %s" % (location))
    #     # Get settings for given location and trigger
    #     SITE, FOUNTAIN, FOLDER = config(location, "Manual")

    #     filename = FOLDER["sim"] + "/DX_sim.csv"

    #     df = pd.read_csv(filename, sep=",")

    #     logger.info(df)
    #     x = df["DX"] * 1000
    #     y = df["Max_IceV"] / df.Max_IceV.max()
    #     y1 = df["Min_T_s"]
        # if location not in ["Schwarzsee 2019"]:
        #     y2 = df["Min_T_c"]
        # ctr = 0

        # for i in range(0, df.shape[0]):
        #     if location == "Guttannen 2021":
        #         lo = ax.scatter(x[i], y[i], marker=".", color=cmap(norm(y1[i])))
        #         if y1[i] > y2[i] and ctr == 0:
        #             ax.scatter(x[i], y[i], s=80, facecolors="none", edgecolors="k")
        #             ax.text(x[i], y[i], str(round(x[i], 0)) + " mm")
        #             ctr = 1

        #     if location == "Guttannen 2020":
        #         lx = ax.scatter(x[i], y[i], marker="+", color=cmap(norm(y1[i])))
        #         if y1[i] > -20 and ctr == 0:
        #             ax.scatter(x[i], y[i], s=80, facecolors="none", edgecolors="k")
        #             ax.text(x[i], y[i], str(round(x[i], 0)) + " mm")
        #             ctr = 1
        #     if location == "Schwarzsee 2019":
        #         lp = ax.scatter(x[i], y[i], marker="x", color=cmap(norm(y1[i])))
        #         if y1[i] > -20 and ctr == 0:
        #             ax.scatter(x[i], y[i], s=80, facecolors="none", edgecolors="k")
        #             ax.text(x[i], y[i], str(round(x[i], 0)) + " mm")
        #             ctr = 1
        # ax.text(x[0], y[0], str(round(df.loc[0, "Max_IceV"], 2)))
        # ax.text(
        #     x[df.index[-1]],
        #     y[df.index[-1]],
        #     str(round(df.loc[df.index[-1], "Max_IceV"], 2)),
        # )
        # ax.set_ylabel("Maximum Ice Volume Sensitivity ($m^3$)")
        # ax.set_xlabel("Ice Layer Thickness ($mm$)")
    # ax.legend(
        # (lo, lx, lp),
        # (locations[0], locations[1], locations[2]),
        # scatterpoints=1,
        # loc="lower right",
        # ncol=1,
        # # fontsize=8
    # )
    # ax.grid()
    # fig.subplots_adjust(right=0.78)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.set_label("Minimum Surface Temperature [$\\degree C$]")
    # pp.savefig(bbox_inches="tight")

    # plt.close()
    # pp.close()
