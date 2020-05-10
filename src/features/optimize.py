import logging
import os
import time
from datetime import datetime
from logging import StreamHandler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
from src.data.config import site, option, folders, fountain, surface
from src.models.air_forecast import fountain_runtime, albedo, projectile_xy, getSEA

from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.colors
import uncertainpy as un
import chaospy as cp

def icestupa(optimize):

    #  read files
    filename0 = os.path.join(folders['input_folder'], site + "_input.csv")
    df = pd.read_csv(filename0, sep=",")
    df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")

    "Simulations"
    fountain['discharge'] = optimize[0]
    fountain['crit_temp'] = optimize[1]
    print(fountain)

    """Constants"""
    Ls = 2848 * 1000  # J/kg Sublimation
    Le = 2514 * 1000  # J/kg Evaporation
    Lf = 334 * 1000  #  J/kg Fusion
    cw = 4.186 * 1000  # J/kg Specific heat water
    ci = 2.108 * 1000  # J/kgC Specific heat ice
    rho_w = 1000  # Density of water
    rho_i = 916  # Density of Ice rho_i
    rho_a = 1.29  # kg/m3 air density at mean sea level
    k = 0.4  # Van Karman constant
    bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
    g = 9.8  # gravity

    """Miscellaneous"""
    time_steps = 5 * 60  # s Model time steps
    p0 = 1013  # Standard air pressure hPa
    theta_f = 45  # Fountain aperture angle
    ftl = 0  # Fountain flight time loss ftl
    dx = 0.001  # Ice layer thickness dx

    """Initialise"""
    start = 0  # model start step
    state = 0
    ice_layer = 0

    l = [
        "T_s",  # Surface Temperature
        "delta_T_s",  # Temperature Change
        "ice",
        "iceV",
        "solid",
        "liquid",
        "vapour",
        "melted",
        "gas",
        "water",
        "sprayed",
        "TotalE",
        "SW",
        "LW",
        "Qs",
        "Ql",
        "meltwater",
        "SA",
        "h_ice",
        "r_ice",
        "SRf",
        "t_droplet",
        "vpa",
        "vp_ice",
        "ppt",
        "theta_s",
        "cld",
    ]
    for col in l:
        df[col] = 0

    """ Estimating Fountain Runtime """
    df["Fountain"] = fountain_runtime(df, fountain)
    if site == 'schwarzsee':
        df.Discharge = df.Discharge * df.Fountain
    else:
        df.Discharge = fountain["discharge"] * df.Fountain

    """ Estimating Albedo """
    df["a"] = albedo(df, surface)


    """ Estimating Fountain Spray radius """
    Area = math.pi * math.pow(fountain["dia_f"], 2) / 4

    for j in range(1, df.shape[0]):
        if option != "schwarzsee":
            df.loc[j, "Discharge"] = fountain["discharge"] * df.loc[j, "Fountain"]
        df.loc[j, "v_f"] = df.loc[j, "Discharge"] / (60 * 1000 * Area)
        df.loc[j, "r_f"], df.loc[j, "t_droplet"] = projectile_xy(
            df.loc[j, "v_f"], theta_f, fountain["h_f"]
        )
    R_f = (
        df["r_f"].replace(0, np.NaN).mean()
    )
    D_t = df["t_droplet"].replace(0, np.NaN).mean()

    """ Simulation """
    for i in range(1, df.shape[0]):

        # Ice Melted
        if (df.loc[i - 1, "ice"] <= 0.05) & (df.Discharge[i:].sum() == 0):
            if df.Discharge[i:].sum() == 0:  # If ice melted after fountain run
                df.loc[i - 1, "solid"] = 0
                df.loc[i - 1, "ice"] = 0
                df.loc[i - 1, "iceV"] = 0
                break

            if start != 0:  # If ice melted in between fountain run
                df.loc[i - 1, "solid"] = 0
                df.loc[i - 1, "ice"] = 0
                df.loc[i - 1, "iceV"] = 0
                state = 0

        # Initiate ice formation
        if (df.loc[i, "Discharge"] > 0) & (start == 0):
            state = 1
            start = i - 1  # Set Model start time
            df.loc[i - 1, "r_ice"] = R_f
            df.loc[i - 1, "h_ice"] = 0

        if state == 1:

            """ Keeping r_ice constant to determine SA """
            if (df.Discharge[i] > 0) & (df.loc[i - 1, "r_ice"] >= R_f):
                # Ice Radius
                df.loc[i, "r_ice"] = df.loc[
                    i - 1, "r_ice"
                ]  # Ice radius same as Initial Fountain Spray Radius

                # Ice Height
                df.loc[i, "h_ice"] = (
                    3 * df.loc[i - 1, "iceV"] / (math.pi * df.loc[i, "r_ice"] ** 2)
                )

                # Height by Radius ratio
                df.loc[i, "h_r"] = df.loc[i - 1, "h_ice"] / df.loc[i - 1, "r_ice"]

                # Area of Conical Ice Surface
                df.loc[i, "SA"] = (
                    math.pi
                    * df.loc[i, "r_ice"]
                    * math.pow(
                        (
                            math.pow(df.loc[i, "r_ice"], 2)
                            + math.pow((df.loc[i, "h_ice"]), 2)
                        ),
                        1 / 2,
                    )
                )

            else:

                """ Keeping h_r constant to determine SA """
                # Height to radius ratio
                df.loc[i, "h_r"] = df.loc[i - 1, "h_r"]

                # Ice Radius
                df.loc[i, "r_ice"] = math.pow(
                    df.loc[i - 1, "iceV"] / math.pi * (3 / df.loc[i, "h_r"]), 1 / 3
                )

                # Ice Height
                df.loc[i, "h_ice"] = df.loc[i, "h_r"] * df.loc[i, "r_ice"]

                # Area of Conical Ice Surface
                df.loc[i, "SA"] = (
                    math.pi
                    * df.loc[i, "r_ice"]
                    * math.pow(
                        (
                            math.pow(df.loc[i, "r_ice"], 2)
                            + math.pow(df.loc[i, "r_ice"] * df.loc[i, "h_r"], 2)
                        ),
                        1 / 2,
                    )
                )


            # Initialize AIR ice layer and update
            if ice_layer == 0:

                ice_layer = dx * df.loc[i, "SA"] * rho_i

                df.loc[i - 1, "ice"] = ice_layer

            else:
                ice_layer = dx * df.loc[i, "SA"] * rho_i


            # Precipitation to ice quantity
            if df.loc[i, "T_a"] < surface["T_rain"]:
                df.loc[i, "ppt"] = (
                    surface["d_ppt"]
                    * df.loc[i, "Prec"]
                    * math.pi
                    * math.pow(df.loc[i, "r_ice"], 2)
                )
                df.loc[i, "solid"] = df.loc[i, "solid"] + df.loc[i, "ppt"]

            # Fountain water output
            df.loc[i, "liquid"] = df.loc[i, "Discharge"] * (1 - ftl) * time_steps / 60

            """ When fountain run """
            if df.loc[i, "liquid"] > 0:

                # Conduction Freezing
                if df.loc[i - 1, "T_s"] < 0:

                    df.loc[i, "solid"] += (ice_layer * ci * (-df.loc[i - 1, "T_s"])) / (
                        Lf
                    )

                    if df.loc[i, "solid"] > df.loc[i, "liquid"]:
                        df.loc[i, "solid"] = df.loc[i, "liquid"]
                        df.loc[i, "liquid"] = 0
                    else:
                        df.loc[i, "liquid"] -= (
                            ice_layer * ci * (-df.loc[i - 1, "T_s"])
                        ) / Lf

                    df.loc[i, "delta_T_s"] = -df.loc[i - 1, "T_s"]

            """ Energy Balance starts """
            # Vapor Pressure empirical relations
            if "vp_a" not in list(df.columns):
                df.loc[i, "vpa"] = (
                        6.11
                        * math.pow(
                    10, 7.5 * df.loc[i - 1, "T_a"] / (df.loc[i - 1, "T_a"] + 237.3)
                )
                        * df.loc[i, "RH"]
                        / 100
                )
            else:
                df.loc[i, "vpa"] = df.loc[i, "vp_a"]

            df.loc[i, "vp_ice"] = 6.112 * np.exp(
                22.46 * (df.loc[i - 1, "T_s"]) / ((df.loc[i - 1, "T_s"]) + 243.12)
            )

            # Sublimation only
            df.loc[i, "Ql"] = (
                0.623
                * Ls
                * rho_a
                / p0
                * math.pow(k, 2)
                * df.loc[i, "v_a"]
                * (df.loc[i, "vpa"] - df.loc[i, "vp_ice"])
                / (
                    np.log(surface["h_aws"] / surface["z0mi"])
                    * np.log(surface["h_aws"] / surface["z0hi"])
                )
            )

            df.loc[i, "gas"] -= (df.loc[i, "Ql"] * (df.loc[i, "SA"]) * time_steps) / Ls

            # Removing gas quantity generated from previous time step
            df.loc[i, "solid"] += (
                df.loc[i, "Ql"] * (df.loc[i, "SA"]) * time_steps
            ) / Ls

            # Ice Temperature
            df.loc[i, "delta_T_s"] += (
                df.loc[i, "Ql"] * (df.loc[i, "SA"]) * time_steps
            ) / (ice_layer * ci)

            """Hot Ice"""
            if df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"] > 0:

                # Melting Ice by Temperature
                df.loc[i, "solid"] -= (
                    (ice_layer * ci)
                    * (-(df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"]))
                    / (-Lf)
                )
                df.loc[i, "melted"] += (
                    (ice_layer * ci)
                    * (-(df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"]))
                    / (-Lf)
                )

                df.loc[i, "delta_T_s"] = -df.loc[i - 1, "T_s"]

            # Estimating Solar Area fraction
            df.loc[i, "theta_s"] = getSEA(
                df.loc[i, "When"],
                fountain["latitude"],
                fountain["longitude"],
                fountain["utc_offset"],
            )
            df.loc[i, "SRf"] = (
                0.5
                * df.loc[i, "h_ice"]
                * df.loc[i, "r_ice"]
                * math.cos(math.radians(df.loc[i, "theta_s"]))
                + math.pi
                * math.pow(df.loc[i, "r_ice"], 2)
                * 0.5
                * math.sin(math.radians(df.loc[i, "theta_s"]))
            ) / (
                math.pi
                * math.pow(
                    (math.pow(df.loc[i, "h_ice"], 2) + math.pow(df.loc[i, "r_ice"], 2)),
                    1 / 2,
                )
                * df.loc[i, "r_ice"]
            )

            # Short Wave Radiation SW
            df.loc[i, "SW"] = (1 - df.loc[i, "a"]) * (
                df.loc[i, "Rad"] * df.loc[i, "SRf"] + df.loc[i, "DRad"]
            )

            # Cloudiness from diffuse fraction
            if df.loc[i, "Rad"] + df.loc[i, "DRad"] > 1:
                df.loc[i, "cld"] = df.loc[i, "DRad"] / (
                    df.loc[i, "Rad"] + df.loc[i, "DRad"]
                )
            else:
                df.loc[i, "cld"] = 0

            # atmospheric emissivity
            df.loc[i, "e_a"] = (
                1.24
                * math.pow(abs(df.loc[i, "vpa"] / (df.loc[i, "T_a"] + 273.15)), 1 / 7)
            ) * (1 + 0.22 * math.pow(df.loc[i, "cld"], 2))

            # Long Wave Radiation LW
            if "oli000z0" not in list(df.columns):

                df.loc[i, "LW"] = df.loc[i, "e_a"] * bc * math.pow(
                    df.loc[i, "T_a"] + 273.15, 4
                ) - surface["ie"] * bc * math.pow(df.loc[i - 1, "T_s"] + 273.15, 4)
            else:
                df.loc[i, "LW"] = df.loc[i, "oli000z0"] - surface["ie"] * bc * math.pow(df.loc[i - 1, "T_s"] + 273.15, 4)

            # Sensible Heat Qs
            df.loc[i, "Qs"] = (
                ci
                * rho_a
                * df.loc[i, "p_a"]
                / p0
                * math.pow(k, 2)
                * df.loc[i, "v_a"]
                * (df.loc[i, "T_a"] - df.loc[i - 1, "T_s"])
                / (
                    np.log(surface["h_aws"] / surface["z0mi"])
                    * np.log(surface["h_aws"] / surface["z0hi"])
                )
            )

            # Total Energy W/m2
            df.loc[i, "TotalE"] = df.loc[i, "SW"] + df.loc[i, "LW"] + df.loc[i, "Qs"]

            # Total Energy Joules
            df.loc[i, "EJoules"] = df.loc[i, "TotalE"] * time_steps * df.loc[i, "SA"]

            if df.loc[i, "EJoules"] < 0:

                """ And fountain on """
                if df.loc[i - 1, "liquid"] > 0:

                    """Freezing water"""

                    df.loc[i, "liquid"] -= (df.loc[i, "EJoules"]) / (-Lf)

                    if df.loc[i, "liquid"] < 0:
                        df.loc[i, "liquid"] += (df.loc[i, "EJoules"]) / (-Lf)
                        df.loc[i, "solid"] += df.loc[i, "liquid"]
                        df.loc[i, "liquid"] = 0
                    else:
                        df.loc[i, "solid"] += (df.loc[i, "EJoules"]) / (-Lf)

                else:
                    """ When fountain off and energy negative """

                    if df.loc[i - 1, "liquid"] < 0:

                        df.loc[i - 1, "liquid"] = 0

                    # Cooling Ice
                    df.loc[i, "delta_T_s"] += (df.loc[i, "EJoules"]) / (ice_layer * ci)


            else:

                # Heating Ice
                df.loc[i, "delta_T_s"] += (df.loc[i, "EJoules"]) / (ice_layer * ci)

                """Hot Ice"""
                if (df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"]) > 0:

                    # Melting Ice by Temperature
                    df.loc[i, "solid"] -= (
                        (ice_layer * ci)
                        * (-(df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"]))
                        / (-Lf)
                    )

                    df.loc[i, "melted"] += (
                        (ice_layer * ci)
                        * (-(df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"]))
                        / (-Lf)
                    )

                    df.loc[i - 1, "T_s"] = 0
                    df.loc[i, "delta_T_s"] = 0



            """ Quantities of all phases """
            df.loc[i, "T_s"] = df.loc[i - 1, "T_s"] + df.loc[i, "delta_T_s"]
            df.loc[i, "meltwater"] = df.loc[i - 1, "meltwater"] + df.loc[i, "melted"]
            df.loc[i, "ice"] = df.loc[i - 1, "ice"] + df.loc[i, "solid"]
            df.loc[i, "vapour"] = df.loc[i - 1, "vapour"] + df.loc[i, "gas"]
            df.loc[i, "sprayed"] = (
                df.loc[i - 1, "sprayed"] + df.loc[i, "Discharge"] * time_steps / 60
            )
            df.loc[i, "water"] = df.loc[i - 1, "water"] + df.loc[i, "liquid"]
            df.loc[i, "iceV"] = df.loc[i, "ice"] / rho_i



    df = df[start:i]

    print("Ice Mass Remaining", float(df["ice"].tail(1)))
    print("Meltwater", float(df["meltwater"].tail(1)))
    print("Ice Volume Max", float(df["iceV"].max()))
    print("Fountain sprayed", float(df["sprayed"].tail(1)))
    print("Ppt", df["ppt"].sum())
    print("Sublimated", float(df["vapour"].tail(1)))
    print("Model ended", df.loc[i - 1, "When"])
    print("Model runtime", df.loc[i - 1, "When"] - df.loc[start, "When"])
    print(
        "Fountain efficiency",
        float((df["meltwater"].tail(1) + df["ice"].tail(1)) / df["sprayed"].tail(1))
        * 100,
    )

    return float(df["iceV"].max()), float((df["meltwater"].tail(1) + df["ice"].tail(1)) / df["sprayed"].tail(1))


problem = {"num_vars": 2, "names": ["discharge", "crit_temp"], "bounds": [[4, 14], [-7, 5]]}

# Generate samples
param_values = saltelli.sample(problem, 10,  calc_second_order=False)

# Output file Initialise
columns = ["discharge", "crit_temp", "Efficiency", "Max IceV"]
index = range(0, param_values.shape[0])
dfo = pd.DataFrame(index=index, columns=columns)
dfo = dfo.fillna(0)
Y = np.zeros([param_values.shape[0]])
Z = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    Y[i], Z[i] = icestupa(X)
    dfo.loc[i, "discharge"] = X[0]
    dfo.loc[i, "crit_temp"] = X[1]
    dfo.loc[i, "Efficiency"] = Z[i]
    dfo.loc[i, "Max IceV"] = Y[i]

dfo = dfo.round(4)
filename2 = os.path.join(
    folders['sim_folder'], site + "_simulations_" + str(problem["names"]) + ".csv"
)
dfo.to_csv(filename2, sep=",")

filename = os.path.join(
    folders['sim_folder'], site + 'salib_maxv' + ".csv"
)
Si = sobol.analyze(problem, Y, print_to_console=True)

Si.to_csv(filename, sep=",")