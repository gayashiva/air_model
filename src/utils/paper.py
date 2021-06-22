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
    logger = logging.getLogger(__name__)
    coloredlogs.install(
        fmt="%(funcName)s %(levelname)s %(message)s",
        level=logging.INFO,
        logger=logger,
    )
    location = "Guttannen 2021"

    # Initialise icestupa object
    icestupa = Icestupa(location)
    SITE, FOLDER = config(location)
    icestupa.read_output()
    icestupa.self_attributes()

    mean_energy = ["SW", "LW", "Ql", "Qs", "Qf", "Qg", "Qt", "Qmelt"]
    for col in mean_energy:
        print("Mean of %s is %.1f\n"%(col,icestupa.df[col].mean()))

    dfd = icestupa.df.set_index("When").resample("D").mean().reset_index()
    dfd["SW_g"] = dfd["SW_direct"] + dfd["SW_diffuse"]
    print("Max of SW_g is %.0f\n"%(dfd["SW_g"].max()))
    print("Max of SW is %.0f\n"%(dfd["SW"].max()))
    for col in mean_energy:
        print("Daily min and max of %s is %.0f, %.0f\n"%(col,dfd[col].min(), dfd[col].max()))

    Total = (
                dfd.SW.abs().sum()
                + dfd.LW.abs().sum()
                + dfd.Qs.abs().sum()
                + dfd.Ql.abs().sum()
                + dfd.Qf.abs().sum()
                + dfd.Qg.abs().sum()
                + dfd.Qmelt.abs().sum()
                + dfd.Qt.abs().sum()
            )
    for col in mean_energy:
        print("Percent of %s is %.0f\n"%(col,dfd[col].abs().sum()/Total*100))

    M_F= round(icestupa.df["Discharge"].sum()* icestupa.DT/ 60 + icestupa.df.loc[0, "input"] - icestupa.V_dome,1)
    M_input = round(icestupa.df["input"].iloc[-1],1)
    M_ppt= round(icestupa.df["ppt"].sum(),1)
    M_dep= round(icestupa.df["dep"].sum(),1)
    M_water = round(icestupa.df["meltwater"].iloc[-1],1)
    M_runoff= round(icestupa.df["unfrozen_water"].iloc[-1],1)
    M_sub = round(icestupa.df["vapour"].iloc[-1],1)
    M_ice = round(icestupa.df["ice"].iloc[-1]- icestupa.V_dome,1)

    print("Contribution of M_F %.1f\n"%(M_F/M_input*100))
    print("Contribution of M_ppt %.1f\n"%(M_ppt/M_input*100))
    print("Contribution of M_dep %.1f\n"%(M_dep/M_input*100))
    print("Storage Efficiency %.0f\n"%(M_water/M_input*100))
    print("Input water %.0f\n"%(M_input))
    print("Dome Volume removed %.0f\n"%(icestupa.V_dome))
