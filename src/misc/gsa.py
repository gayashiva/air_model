
import sys
import os
import numpy as np
import pandas as pd
import math
import matplotlib.colors
import uncertainpy as un
import statistics as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

if __name__ == "__main__":

    locations = ['guttannen21', 'guttannen20', 'gangles21']

    for i,location in enumerate(locations):
        SITE, FOLDER = config(location)
        icestupa = Icestupa(location)
        icestupa.read_output()
        icestupa.self_attributes()

        variance = []
        mean = []
        evaluations = []

        data = un.Data()
        filename1 = FOLDER['sim']+ "SE_full.h5"
        # filename1 = FOLDER['sim']+ "full.h5"
        # filename1 = FOLDER['sim']+ "efficiency.h5"
        data.load(filename1)
        # print(data)
        print(location)
        params = ['IE', 'A_I', 'A_S','A_DECAY', 'T_PPT', 'Z', 'DX']
        print(params)
        data = data[location]
        print(data['sobol_total_average'])
