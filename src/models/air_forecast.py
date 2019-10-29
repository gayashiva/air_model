import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
import math
import sys
import logging
from src.data.config import fountain, materials, weather, aws

np.seterr(all="raise")

def albedo(df, materials):

    materials['t_d'] = materials['t_d'] * 24 * 60 / 5  # convert to 5 minute time steps
    ti = materials['t_d']
    tw = materials['t_d']
    s = 0  # Initialised
    w = 0
    j = 0  # Account for decay rate after rain
    rf = 2  # Rain decay factor
    Ts = -1  # Solid Ppt

    for i in range(1, df.shape[0]):

        # Wind Sensor Snow covered and fountain off
        if (df.loc[i, "v_a"] == 0) & (df.loc[i, "Discharge"] == 0):
            if df.loc[i, "Prec"] > 0 & (
                df.loc[i, "T_a"] < Ts
            ):  # Assumes snow ppt because of wind sensor
                s = 0
                w = 0
                j = 0
                tw = materials['t_d']  # Decay rate reset after snowfall
            df.loc[i, "a"] = materials['a_i'] + (materials['a_s'] - materials['a_i']) * math.exp(-s / ti)
            s = s + 1

        # No snow and fountain off
        if (df.loc[i, "v_a"] != 0) & (df.loc[i, "Discharge"] == 0):
            df.loc[i, "a"] = materials['a_w'] + (materials['a_i'] - materials['a_w']) * math.exp(-w / tw)
            w = w + 1

        # Fountain on
        if df.loc[i, "Discharge"] > 0:
            df.loc[i, "a"] = materials['a_w']
            w = 0
            s = 0
            j = 0
            tw = materials['t_d']  # Decay rate reset after fountain

        # Liquid Ppt
        if (df.loc[i, "Prec"] > 0) & (df.loc[i, "T_a"] > Ts):
            if j == 0:
                tw = tw / rf  # Decay rate speeds up after rain
                j = 1
            df.loc[i, "a"] = materials['a_w']

    return df["a"]


def projectile_xy(v, theta_f, hs=0.0, g=9.8):
    """
    calculate a list of (x, y) projectile motion data points
    where:
    x axis is distance (or range) in meters
    y axis is height in meters
    v is muzzle velocity of the projectile (meter/second)
    theta_f is the firing angle with repsect to ground (radians)
    hs is starting height with respect to ground (meters)
    g is the gravitational pull (meters/second_square)
    """
    data_xy = []
    t = 0.0
    while True:
        # now calculate the height y
        y = hs + (t * v * math.sin(theta_f)) - (g * t * t) / 2
        # projectile has hit ground level
        if y < 0:
            break
        # calculate the distance x
        x = v * math.cos(theta_f) * t
        # append the (x, y) tuple to the list
        data_xy.append((x, y))
        # use the time in increments of 0.1 seconds
        t += 0.01
    return max(data_xy)[0], t


def icestupa(
    df,
    fountain,
    weather,
    aws,
    materials,
):

    logger = logging.getLogger(__name__)
    logger.debug('This is a debug message')
    logger.info('This is for temp')
    logger.warning('This is for solid')
    logger.error('This is for melted')
    logger.critical('This is a critical message')

    """Material Properties"""
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

    """Initialise"""
    df["T_s"] = 0  # Surface Temperature
    df["temp"] = 0  # Temperature Change
    df["e_s"] = materials['ie']  # Surface Emissivity
    df["ice"] = 0
    df["iceV"] = 0
    df["solid"] = 0
    df["liquid"] = 0
    df["vapour"] = 0
    df["melted"] = 0
    df["gas"] = 0
    df["water"] = 0
    df["TotalE"] = 0
    df["SW"] = 0
    df["LW"] = 0
    df["Qs"] = 0
    df["Ql"] = 0
    df["meltwater"] = 0
    df["SA"] = 0  # Surface Area
    df["h_ice"] = 0
    df["h_f"] = fountain['h_f']
    y = fountain['h_f']
    df["d_t"] = 0
    prec = 0  # Precipitation
    start = 0  # model start step
    stop = 0  # model end step
    state = 0
    df["SRf"] = 0
    ice_layer = 0
    T_droplet = 0

    fountain['theta_f'] = math.radians(fountain['theta_f'])  # Angle of Spray
    weather['theta_s'] = math.radians(weather['theta_s'])  # solar angle

    """ Estimating Albedo """
    df["a"] = albedo(df, materials)

    """ Estimating Fountain Spray radius """
    Area = 3.14 * math.pow(fountain['d_f'], 2) / 4
    df["v"] = 0
    df["r"] = 0

    for j in range(1, df.shape[0]):
        df.loc[j, "v_f"] = df.loc[j, "Discharge"] / (60 * 1000 * Area)
        df.loc[j, "r_f"], df.loc[j, "d_t"] = projectile_xy(
            df.loc[j, "v_f"], fountain['theta_f'], fountain['h_f']
        )
    R_f = df["r_f"].replace(0, np.NaN).mean()
    D_t = df["d_t"].replace(0, np.NaN).mean()

    """ Simulation """
    for i in range(1, df.shape[0]):

        if (df.loc[i - 1, "ice"] <= 0) & (df.Discharge[i:].sum() == 0) :
            df.loc[i - 1, "solid"] = 0
            df.loc[i - 1, "ice"] = 0
            df.loc[i - 1, "iceV"] = 0
            stop = i - 1
            break

        if (df.loc[i, "Discharge"]) > 0:
            state = 1
            # Set Model start time
            if start == 0:
                start = i - 1

        if state == 1:

            if (
                df.Discharge[i:].sum() > 0
            ):  # Keeping Radius for the given fountain parameters

                # Ice radius same as Fountain Spray Radius
                df.loc[i, "r_ice"] = R_f

                # Ice Height
                df.loc[i, "h_ice"] = (
                    3 * df.loc[i - 1, "iceV"] / (math.pi * df.loc[i, "r_ice"] ** 2)
                )

                df.loc[i, "h_r"] = (
                    df.loc[i, "h_ice"] / df.loc[i, "r_ice"]
                )  # Height to radius ratio

                df.loc[i, "SRf"] = (
                    0.5
                    * (df.loc[i, "h_r"] + math.pi)
                    / (
                        math.pi
                        * math.pow((1 + math.pow(df.loc[i, "h_r"], 2)), 1 / 2)
                        * math.pow(2, 1 / 2)
                    )
                )  # Estimating Solar Area fraction if theta_s 45

                if df.loc[i, "h_f"] > df.loc[i, "h_ice"]:  # Fountain height Check

                    df.loc[i, "SA"] = (
                        math.pi
                        * df.loc[i, "r_ice"]
                        * math.pow(
                            (
                                math.pow(df.loc[i, "r_ice"], 2)
                                + math.pow(df.loc[i, "h_ice"], 2)
                            ),
                            1 / 2,
                        )
                    )  # Area of Conical Ice Surface

                else:

                    y = y + fountain['h_f']
                    df.loc[i:, "h_f"] = y  # Reset height

                    for j in range(1, df.shape[0]):
                        df.loc[j, "v_f"] = df.loc[j, "Discharge"] / (60 * 1000 * Area)
                        df.loc[j, "r_f"], df.loc[j, "d_t"] = projectile_xy(
                            df.loc[j, "v_f"], fountain['theta_f'], df.loc[i, "h_f"]
                        )
                    R_f = df["r_f"].replace(0, np.NaN).mean()
                    D_t = df["d_t"].replace(0, np.NaN).mean()

                    df.loc[i, "r_ice"] = R_f

                    df.loc[i, "SA"] = (
                        math.pi
                        * df.loc[i, "r_ice"]
                        * math.pow(
                            (
                                math.pow(df.loc[i, "r_ice"], 2)
                                + math.pow(df.loc[i, "h_ice"], 2)
                            ),
                            1 / 2,
                        )
                    )  # Area of Conical Ice Surface

            else:  # Keeping h_r constant to determine SA

                df.loc[i, "h_r"] = (
                    df.loc[i - 1, "h_ice"] / df.loc[i - 1, "r_ice"]
                )  # Height to radius ratio

                # Ice Radius
                df.loc[i, "r_ice"] = math.pow(
                    df.loc[i - 1, "iceV"] / math.pi * (3 / df.loc[i, "h_r"]), 1 / 3
                )  # Assumption height h_r and conical

                df.loc[i, "h_ice"] = df.loc[i, "h_r"] * df.loc[i, "r_ice"]

                # Estimating Solar Area fraction if theta_s 45
                df.loc[i, "SRf"] = (
                    0.5
                    * (df.loc[i, "h_r"] + math.pi)
                    / (
                        math.pi
                        * math.pow((1 + math.pow(df.loc[i, "h_r"], 2)), 1 / 2)
                        * math.pow(2, 1 / 2)
                    )
                )

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
                )  # Area of Conical Ice Surface

            # Update Ice Layer
            if (ice_layer != 0) & (df.loc[i - 1, "SA"] != df.loc[i, "SA"]):
                ice_layer = (
                    materials['dx'] * df.loc[i, "SA"] * rho_i
                )
                logger.info('Ice layer is %s thick at %s', ice_layer, df.loc[i, "When"])

            # Precipitation to ice quantity
            prec = prec + weather['dp'] * df.loc[i, "Prec"] * math.pi * math.pow(R_f, 2)
            df.loc[i - 1, "ice"] = df.loc[i - 1, "ice"] + weather['dp'] * df.loc[
                i, "Prec"
            ] * math.pi * math.pow(R_f, 2)

            # Vapor Pressure empirical relations
            if "vp_a" not in list(df.columns):
                Ea = (
                    6.11
                    * math.pow(
                        10, 7.5 * df.loc[i - 1, "T_a"] / (df.loc[i - 1, "T_a"] + 237.3)
                    )
                    * df.loc[i, "RH"]
                    / 100
                )
            else:
                Ea = df.loc[i, "vp_a"]

            Ew = 6.112 * np.exp(17.62 * fountain['T_f'] / (fountain['T_f'] + 243.12))
            Eice = 6.112 * np.exp(
                22.46 * (df.loc[i - 1, "T_s"]) / ((df.loc[i - 1, "T_s"]) + 243.12)
            )

            # atmospheric emissivity
            if df.loc[i, "Prec"] > 0:  # c = 1
                df.loc[i, "e_a"] = (
                    1.24 * math.pow(abs(Ea / (df.loc[i, "T_a"] + 273.15)), 1 / 7) * 1.22
                )
            else:
                df.loc[i, "e_a"] = (
                    1.24
                    * math.pow(abs(Ea / (df.loc[i, "T_a"] + 273.15)), 1 / 7)
                    * (1 + 0.22 * math.pow(weather['c'], 2))
                )

            # Fountain water output
            df.loc[i, "liquid"] = df.loc[i, "Discharge"] * (1 - fountain['ftl']) * aws['time_steps'] / (60)

            """ When fountain run """
            if df.loc[i, "liquid"] > 0:

                # Initialize AIR ice layer
                if ice_layer == 0:

                    ice_layer = (
                        materials['dx'] * df.loc[i, "SA"] * rho_i
                    )
                    logger.warning('Ice layer is %s thick at %s', ice_layer, df.loc[i, "When"])

                    if ice_layer == 0 :
                        logger.critical('Ice layer is %s thick at %s', ice_layer, df.loc[i, "SA"], df.loc[i, "When"])
                        break
                    df.loc[i - 1, "ice"] = ice_layer

                if df.loc[ i-1 , "T_s"] < 0 :

                    # Initial freeze up due to ice layer
                    df.loc[i,'solid'] += (ice_layer * ci * (-df.loc[ i-1, "T_s"])) / (
                        Lf
                    )

                    df.loc[i,'liquid'] -=(ice_layer * ci * (-df.loc[ i-1, "T_s"])) / (
                        Lf
                    )

                    logger.error('Ice layer made %s thick ice at %s', df.loc[i,'solid'], df.loc[i, "When"])
                    df.loc[i, "temp"] = -df.loc[i-1, "T_s"]

                # Evaporation or condensation
                df.loc[i, "Ql"] = (
                    0.623
                    * Le
                    * rho_a
                    / materials['p0']
                    * math.pow(k, 2)
                    * df.loc[i, "v_a"]
                    * (Ea - Ew)
                    / (np.log(aws['z'] / weather['z0mi']) * np.log(aws['z'] / weather['z0hi']))
                )

                df.loc[i, "gas"] -= (
                    df.loc[i, "Ql"] * df.loc[i, "SA"] * aws['time_steps']
                ) / Le

                # Removing water quantity generated from previous time step
                df.loc[i, "liquid"] += (
                    df.loc[i, "Ql"] * df.loc[i, "SA"] * aws['time_steps']
                ) / Le

                df.loc[i, "e_s"] = materials['we']

            else:  # When fountain off

                if df.loc[i - 1, "ice"] > 0:

                    # Initialise with zero value
                    df.loc[i, "e_s"] = materials['ie']

                    # Sublimation, Evaporation or condensation
                    df.loc[i, "Ql"] = (
                        0.623
                        * Ls
                        * rho_a
                        / materials['p0']
                        * math.pow(k, 2)
                        * df.loc[i, "v_a"]
                        * (Ea - Eice)
                        / (np.log(aws['z'] / weather['z0mi']) * np.log(aws['z'] / weather['z0hi']))
                    )

                    df.loc[i, "gas"] -= (
                        df.loc[i, "Ql"] * (df.loc[i, "SA"]) * aws['time_steps']
                    ) / Ls

                    # Removing gas quantity generated from previous time step
                    df.loc[i - 1, "ice"] += (
                        df.loc[i, "Ql"] * (df.loc[i, "SA"]) * aws['time_steps']
                    ) / Ls

                    # Ice Temperature
                    df.loc[i, "temp"] += (
                        df.loc[i, "Ql"] * (df.loc[i, "SA"]) * aws['time_steps']
                    ) / (ice_layer * ci)

                    """Hot Ice"""
                    if df.loc[i - 1, "T_s"] + df.loc[i, "temp"] > 0:

                        # Melting Ice by Temperature
                        df.loc[i, "solid"] -= (
                            (ice_layer * ci)
                            * (-(df.loc[i - 1, "T_s"] + df.loc[i, "temp"]))
                            / (-Lf)
                        )
                        df.loc[i, "melted"] += (
                            (ice_layer * ci)
                            * (-(df.loc[i - 1, "T_s"] + df.loc[i, "temp"]))
                            / (-Lf)
                        )

                        df.loc[i, "temp"] = -df.loc[i - 1, "T_s"]

            logger.info('Ice made after sublimation is %s thick at %s', round(df.loc[i, "solid"]), df.loc[i, "When"])

            # Short Wave Radiation SW
            df.loc[i, "SW"] = (1 - df.loc[i, "a"]) * (
                df.loc[i, "Rad"] * df.loc[i, "SRf"] + df.loc[i, "DRad"]
            )

            # Long Wave Radiation LW
            if "oli000z0" not in list(df.columns):

                df.loc[i, "LW"] = df.loc[i, "e_a"] * bc * math.pow(
                    df.loc[i, "T_a"] + 273.15, 4
                ) - df.loc[i, "e_s"] * bc * math.pow(df.loc[i - 1, "T_s"] + 273.15, 4)
            else:
                df.loc[i, "LW"] = df.loc[i, "oli000z0"] - df.loc[
                    i, "e_s"
                ] * bc * math.pow(df.loc[i - 1, "T_s"] + 273.15, 4)

            # Sensible Heat Qs
            df.loc[i, "Qs"] = (
                ci
                * rho_a
                * df.loc[i, "p_a"]
                / materials['p0']
                * math.pow(k, 2)
                * df.loc[i, "v_a"]
                * (df.loc[i, "T_a"] - df.loc[i - 1, "T_s"])
                / (np.log(aws['z'] / weather['z0mi']) * np.log(aws['z'] / weather['z0hi']))
            )

            # Total Energy W/m2
            df.loc[i, "TotalE"] = df.loc[i, "SW"] + df.loc[i, "LW"] + df.loc[i, "Qs"]

            # Total Energy Joules
            df.loc[i, "EJoules"] = df.loc[i, "TotalE"] * aws['time_steps'] * df.loc[i, "SA"]

            if df.loc[i - 1, "ice"] > 0:

                if df.loc[i, "EJoules"] < 0:  # Energy Negative

                    if df.loc[i - 1, "liquid"] > 0:
                        """Freezing water"""

                        df.loc[i, "liquid"] -= (df.loc[i, "EJoules"]) / (
                            -Lf
                        )

                        if df.loc[i, "liquid"] < 0:
                            df.loc[i, "liquid"] += (df.loc[i, "EJoules"]) / (
                                -Lf
                            )
                            df.loc[i, "solid"] += df.loc[i, "liquid"]
                            df.loc[i, "liquid"] = 0
                        else:
                            df.loc[i, "solid"] += (df.loc[i, "EJoules"]) / (
                                -Lf
                            )

                        logger.warning('Ice made after energy neg is %s thick at temp %s', round(df.loc[i, "solid"]), df.loc[i - 1, "T_s"])

                    else: # When fountain off and energy negative

                        if df.loc[i - 1, "liquid"] < 0:
                            logger.critical(
                                'Liquid is %s at %s',
                                round(df.loc[i, "temp"]),
                                df.loc[i, "When"],
                            )
                            df.loc[i - 1, "liquid"] = 0


                        # Cooling Ice
                        df.loc[i, "temp"] += (df.loc[i, "EJoules"]) / (ice_layer * ci)


                    logger.info('Ice made after energy neg is %s thick at %s', round(df.loc[i, "solid"]), df.loc[i, "When"])

                else:  # Energy Positive

                    if df.loc[i - 1, "ice"] > 0:

                        # Heating Ice
                        df.loc[i, "temp"] += (df.loc[i, "EJoules"]) / (ice_layer * ci)

                        """Hot Ice"""
                        if (df.loc[i - 1, "T_s"] + df.loc[i, "temp"]) > 0:

                            # Melting Ice by Temperature
                            df.loc[i, "solid"] -= (
                                (ice_layer * ci)
                                * (-(df.loc[i - 1, "T_s"] + df.loc[i, "temp"]))
                                / (-Lf)
                            )

                            df.loc[i, "melted"] += (
                                (ice_layer * ci)
                                * (-(df.loc[i - 1, "T_s"] + df.loc[i, "temp"]))
                                / (-Lf)
                            )

                            df.loc[i - 1, "T_s"] = 0
                            df.loc[i, "temp"] = 0

                        logger.info('Ice melted is %s thick at %s', round(df.loc[i, "solid"]), df.loc[i, "When"])

            if df.loc[i, "temp"] < -50:
                logger.critical(
                    'Temperature change is %s at %s',
                    round(df.loc[i, "temp"]),
                    df.loc[i, "When"],
                )

            """ Quantities of all phases """
            df.loc[i, "T_s"] = df.loc[i - 1, "T_s"] + df.loc[i, "temp"]
            df.loc[i, "water"] = df.loc[i - 1, "water"] + df.loc[i, "liquid"]
            df.loc[i, "meltwater"] = df.loc[i - 1, "meltwater"] + df.loc[i, "melted"]
            df.loc[i, "ice"] = df.loc[i - 1, "ice"] + df.loc[i, "solid"]
            df.loc[i, "vapour"] = df.loc[i - 1, "vapour"] + df.loc[i, "gas"]
            df.loc[i, "iceV"] = df.loc[i, "ice"] / rho_i

            logger.debug('Surface temp. %s, is Ice is %s at %s', round(df.loc[i, "T_s"]), round(df.loc[i, "ice"]), df.loc[i, "When"])

    df = df[start:i]

    print("Ice Mass Remaining", float(df["ice"].tail(1)))
    print("Meltwater", float(df["meltwater"].tail(1)))
    print("Evaporated/sublimated", float(df["vapour"].tail(1)))
    print("Ice Volume Max", float(df["iceV"].max()))
    print("Ppt", prec)
    print("Model ended", df.loc[i - 1, "When"])
    print("Model runtime", df.loc[i - 1, "When"] - df.loc[start, "When"])

    return df
