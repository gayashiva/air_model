"""Icestupa class function that generates figures for web app
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, sys
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa

if __name__ == "__main__":
    # locations = ["gangles21", "guttannen21"]
    locations = [ "gangles21"]

    fig, ax = plt.subplots()
    for ctr, location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.df = icestupa.df[:icestupa.last_hour]

        df = icestupa.df[
            [
                "Qs",
                "Ql",
                "SW",
                "LW",
                "Qf",
                "Qg",
                "SA",
                "T_a",
                "v_a",
                "RH",
                "Discharge",
                # "T_s",
                # "Qt",
                # "sea",
                # "s_cone",
            ]
        ]
        corr = df.corr()
        print(corr)
        for col1 in df.columns:
            for col2 in df.columns:
                if abs(corr.loc[col1,col2]) < 0.5:
                    corr.loc[col1,col2] = 0
                if col1 == col2:
                    corr.loc[col1,col2] = 0
        # corr = corr.style.background_gradient(cmap='coolwarm').set_precision(2)
        # ax.matshow(corr)
        # sns.heatmap(corr, annot=True, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
        #             square=True, ax=ax)
        # matrix = np.triu(corr)
        sns.heatmap(corr, annot=False, vmin=-0.55, vmax=0.55, center= 0, cmap= 'coolwarm', cbar=False,  ax=ax)
        # plt.xticks(range(len(corr.columns)), corr.columns)
        # plt.yticks(range(len(corr.columns)), corr.columns)
        plt.savefig(
            "data/paper/corr_matrix_" + location+".jpg",
            dpi=300,
            bbox_inches="tight",
        )
# Create correlation matrix
        corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] < 0.5)]

        print(to_drop)
# Drop features 
        # df.drop(to_drop, axis=1, inplace=True)
