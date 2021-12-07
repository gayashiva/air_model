"""Analyse GSA results to decide most sensitive parameters"""
import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import uncertainpy as un
import statistics as st
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

    locations = ["gangles21", "guttannen21"]
    # locations = [ 'gangles21']

    for i, location in enumerate(locations):
        print(f'\n\tLocation is {location} ')
        CONSTANTS, SITE, FOLDER = config(location)

        data = un.Data()
        filename1 = FOLDER["sim"] + "globalSA.h5"

        data.load(filename1)
        data1 = data[location]

        for param, value in zip(data.uncertain_parameters,data1['sobol_total_average'] ):
            print(f'\t{param} has total order sens. =  {round(value,2)}')

