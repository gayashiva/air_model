"""Analyse GSA results to decide most sensitive parameters"""
import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import uncertainpy as un
import statistics as st
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.offsetbox import AnchoredText
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, coloredlogs

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    locations = ["guttannen21", "gangles21"]
    # locations = [ 'guttannen21']

    result = []
    for i, location in enumerate(locations):
        print(f'\n\tLocation is {location} ')
        CONSTANTS, SITE, FOLDER = config(location)

        data = un.Data()
        filename1 = FOLDER["sim"] + "globalSA.h5"

        data.load(filename1)
        data1 = data[location]
        # print(data)

        for param, value in zip(data.uncertain_parameters,data1['sobol_total_average'] ):
            print(f'\t{param} has total order sens. =  {round(value,2)}')
            result.append(
                [
                    get_parameter_metadata(location)["shortname"],
                    get_parameter_metadata(param)["latex"],
                    round(value,3),
                ]
            )
        if location == 'guttannen21':
            result.append(
                [
                    get_parameter_metadata('guttannen21')["shortname"],
                    get_parameter_metadata('D_F')["latex"],
                    0.05, #From GSA2
                ]
            )

    df = pd.DataFrame(result, columns=["AIR", "param", "value"])
    print(df)
    df.to_csv("data/paper1/GSA.csv")

    print(df)
    fig, ax = plt.subplots()
    ax = sns.barplot(
        y="param", x="value", hue="AIR", data=df, palette="Set1"
    )
    ax.set_ylabel("Parameter")
    ax.set_xlabel("Sensitivity of Net Water Loss")
    plt.savefig("data/paper1/sensitivities.jpg", bbox_inches="tight", dpi=300)
    plt.clf()
