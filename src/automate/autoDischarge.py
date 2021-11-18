import math
import numpy as np
from pvlib import location, atmosphere
from datetime import datetime
from projectile import get_projectile
import json
import logging
import coloredlogs

import os, sys
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(dirname)
from src.utils.settings import config

"""Physical Constants"""
DT = 60 * 60  # Model time step
L_S = 2848 * 1000  # J/kg Sublimation
L_F = 334 * 1000  # J/kg Fusion
C_I = 2.097 * 1000  # J/kgC Specific heat ice
C_W = 4.186 * 1000  # J/kgC Specific heat water
C_A=1010  # specific heat of air [J kg-1 K-1]
RHO_A=1.29  # air density at mean sea level
RHO_W = 1000  # Density of water
RHO_I = 917  # Density of Ice RHO_I
VAN_KARMAN = 0.4  # Van Karman constant
K_I = 2.123  # Thermal Conductivity Waite et al. 2006
STEFAN_BOLTZMAN = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
P0 = 1013  # Standard air pressure hPa
G = 9.81  # Gravitational acceleration

"""Surface Properties"""
IE = 0.97  # Ice Emissivity IE
Z = 0.003  # Ice Momentum and Scalar roughness length
DX = 20e-03  # m Surface layer thickness growth rate
H_AWS = 2

def Automate(aws, site="guttannen21", virtual_r = 0):

    CONSTANTS, SITE, FOLDER = config(site)

    # with open("data/" + site + "/info.json") as f:
    with open(FOLDER["raw"] + "info.json") as f:
        params = json.load(f)

    # AWS
    temp = aws[0]
    rh = aws[1]
    wind = aws[2]

    # Derived
    press = atmosphere.alt2pres(params["alt"]) / 100

    vp_a = (
        6.107
        * math.pow(
            10,
            7.5 * temp / (temp + 237.3),
        )
        * rh
        / 100
    )

    vp_ice = np.exp(43.494 - 6545.8 / (params["temp_i"] + 278)) / ((params["temp_i"] + 868) ** 2 * 100)

    e_a = (1.24 * math.pow(abs(vp_a / (temp + 273.15)), 1 / 7)) * (
        1 + 0.22 * math.pow(params["cld"], 2)
    )

    LW = e_a * STEFAN_BOLTZMAN * math.pow(
        temp + 273.15, 4
    ) - IE * STEFAN_BOLTZMAN * math.pow(273.15 + params["temp_i"], 4)

    Qs = (
        C_A
        * RHO_A
        * press
        / P0
        * math.pow(VAN_KARMAN, 2)
        * wind
        * (temp - params["temp_i"])
        / ((np.log(H_AWS / Z)) ** 2)
    )

    Ql = (
        0.623
        * L_S
        * RHO_A
        / P0
        * math.pow(VAN_KARMAN, 2)
        * wind
        * (vp_a - vp_ice)
        / ((np.log(H_AWS / Z)) ** 2)
    )

    Qf = (
        RHO_I
        * DX
        * C_I
        / DT
        * params["temp_i"]
    )

    freezing_energy = Ql + Qs + LW + Qf
    dis = -1 * freezing_energy / L_F * 1000 / 60

    if virtual_r:
        SA = math.pi * math.pow(params["virtual_r"],2)
        dis *= SA

    return dis

