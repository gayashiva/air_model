"""Function that generates csv tables for the paper
"""
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import logging, os, sys,coloredlogs
from codetiming import Timer
import csv

# Locals
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa
from src.utils.settings import config
from src.utils import setup_logger

if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # answers = dict(
    #     # location="Schwarzsee 2019",
    #     location="Guttannen 2020",
    #     # location="Gangles 2021",
    # )
    locations = ["Guttannen 2021", "Guttannen 2020", "Gangles 2021"]
    filenames = []

    for location in locations:
        # Initialise icestupa object
        icestupa = Icestupa(location)
        CONSTANTS, SITE, FOLDER = config(location)
        icestupa.read_output()
        icestupa.self_attributes()

#         M_F= round(icestupa.df["Discharge"].sum()* icestupa.DT/ 60 + icestupa.df.loc[0, "input"] - icestupa.V_dome *
#             icestupa.RHO_I,1)
#         M_input = round(icestupa.df["input"].iloc[-1],1)
#         M_ppt= round(icestupa.df["ppt"].sum(),1)
#         M_dep= round(icestupa.df["dep"].sum(),1)
#         M_water = round(icestupa.df["meltwater"].iloc[-1],1)
#         M_runoff= round(icestupa.df["unfrozen_water"].iloc[-1],1)
#         M_sub = round(icestupa.df["vapour"].iloc[-1],1)
#         M_ice = round(icestupa.df["ice"].iloc[-1]- icestupa.V_dome * icestupa.RHO_I,1)
#         Mass_Component = location
#         var_dict={}
#         for var in ["Mass_Component", "M_F", "M_ppt", "M_dep", "M_ice", "M_sub", "M_water", "M_runoff"]:
#             var_dict[var] = eval(var)
#         print(var_dict)
#         a_file = open(FOLDER["output"] + "mass_bal.csv", "w")
#         writer = csv.writer(a_file)
#         for key, value in var_dict.items():
#             # key = '$' + key + '$'
#             key = key[2:]
#             writer.writerow([key, value])
#             print([key, value])

        filenames.append(FOLDER["output"] + "mass_bal.csv")

    # merging csv files
    df = pd.concat(map(pd.read_csv, filenames), ignore_index=False)
    print(df.columns)

    df = df.rename(
        {
            "Guttannen 2021": "CH21",
            "Guttannen 2020": "CH20",
            "Gangles 2021": "IN21",
            "Schwarzsee 2019": "CH19",
            "ss_Component": "Mass",
        },
        axis=1,
    )
    df = df.set_index("Mass")
    df = df.groupby(level=0).sum().reset_index()
    # print(mass_table)

    for i in range(0,df.shape[0]):
        print(df.loc[i,"Mass"])
        df.loc[i,"Mass"] = '$M_{' +df.loc[i,"Mass"] + '}$'

    df = df.set_index("Mass")
    df.to_csv("data/paper1/mass_bal.csv")
    print(df)


