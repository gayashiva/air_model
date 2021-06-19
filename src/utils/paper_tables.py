"""Function that generates csv tables for the paper
"""
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import logging, os, sys,coloredlogs
from codetiming import Timer

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

    answers = dict(
        # location="Schwarzsee 2019",
        location="Guttannen 2021",
        # location="Gangles 2021",
    )

    # Initialise icestupa object
    icestupa = Icestupa(answers["location"])
    icestupa.read_output()
    M_F= round(icestupa.df["Discharge"].sum()* icestupa.DT/ 60,1)
    M_input = round(icestupa.df["input"].iloc[-1],1)
    M_ppt= round(icestupa.df["ppt"].sum(),1)
    M_dep= round(icestupa.df["dep"].sum(),1)
    M_water = round(icestupa.df["meltwater"].iloc[-1],1)
    M_runoff= round(icestupa.df["unfrozen_water"].iloc[-1],1)
    M_sub = round(icestupa.df["vapour"].iloc[-1],1)
    # M_ice = round((icestupa.df["iceV"].iloc[-1]-icestupa.dome_vol)/icestupa.RHO_I,1)
    M_ice = round(icestupa.df["ice"].iloc[-1]-icestupa.dome_vol*icestupa.RHO_I,1)
    print(M_ice, M_sub, M_water)
    print(M_F+M_ppt+M_dep)
    print(M_ice+M_sub+M_water+M_runoff)
    print(M_input)
    print(icestupa.df.loc[0, "iceV"], icestupa.df.loc[0, "input"]/icestupa.RHO_I,icestupa.dome_vol)


