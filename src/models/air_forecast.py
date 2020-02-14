import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import datetime
import logging
from src.data.config import fountain, surface, site, option, dates, folders

pd.options.mode.chained_assignment = None  # Suppress Setting with warning


# def albedo(df, surface):
#
#     surface["decay_t"] = (
#         surface["decay_t"] * 24 * 60 / 5
#     )  # convert to 5 minute time steps
#     s = 0
#     f = 0
#
#     for i in range(1, df.shape[0]):
#
#         ti = surface["decay_t"]
#         a_min = surface["a_i"]
#
#         # Precipitation
#         if (df.loc[i, "Fountain"] == 0) & (df.loc[i, "Prec"] > 0):
#             if df.loc[i, "T_a"] < surface["rain_temp"]:  # Snow
#                 s = 0
#                 f = 0
#
#         if df.loc[i, "Fountain"] > 0:
#             f = 1
#             s = 0
#
#         if f == 0:  # last snowed
#             df.loc[i, "a"] = a_min + (surface["a_s"] - a_min) * math.exp(-s / ti)
#             s = s + 1
#         else:  # last sprayed
#             df.loc[i, "a"] = a_min
#             s = s + 1
#
#     return df["a"]
#
# def projectile_xy(v, theta_f, hs=0.0, g=9.8):
#     """
#     calculate a list of (x, y) projectile motion data points
#     where:
#     x axis is distance (or range) in meters
#     y axis is height in meters
#     v is muzzle velocity of the projectile (meter/second)
#     theta_f is the firing angle with repsect to ground (degrees)
#     hs is starting height with respect to ground (meters)
#     g is the gravitational pull (meters/second_square)
#     """
#     data_xy = []
#     t = 0.0
#     theta_f = math.radians(theta_f)
#     while True:
#         # now calculate the height y
#         y = hs + (t * v * math.sin(theta_f)) - (g * t * t) / 2
#         # projectile has hit ground level
#         if y < 0:
#             break
#         # calculate the distance x
#         x = v * math.cos(theta_f) * t
#         # append the (x, y) tuple to the list
#         data_xy.append((x, y))
#         # use the time in increments of 0.1 seconds
#         t += 0.01
#     return x, t
#
# def getSEA(date, latitude, longitude, utc_offset):
#     hour = date.hour
#     minute = date.minute
#     # Check your timezone to add the offset
#     hour_minute = (hour + minute / 60) - utc_offset
#     day_of_year = date.timetuple().tm_yday
#
#     g = (360 / 365.25) * (day_of_year + hour_minute / 24)
#
#     g_radians = math.radians(g)
#
#     declination = (
#         0.396372
#         - 22.91327 * math.cos(g_radians)
#         + 4.02543 * math.sin(g_radians)
#         - 0.387205 * math.cos(2 * g_radians)
#         + 0.051967 * math.sin(2 * g_radians)
#         - 0.154527 * math.cos(3 * g_radians)
#         + 0.084798 * math.sin(3 * g_radians)
#     )
#
#     time_correction = (
#         0.004297
#         + 0.107029 * math.cos(g_radians)
#         - 1.837877 * math.sin(g_radians)
#         - 0.837378 * math.cos(2 * g_radians)
#         - 2.340475 * math.sin(2 * g_radians)
#     )
#
#     SHA = (hour_minute - 12) * 15 + longitude + time_correction
#
#     if SHA > 180:
#         SHA_corrected = SHA - 360
#     elif SHA < -180:
#         SHA_corrected = SHA + 360
#     else:
#         SHA_corrected = SHA
#
#     lat_radians = math.radians(latitude)
#     d_radians = math.radians(declination)
#     SHA_radians = math.radians(SHA)
#
#     SZA_radians = math.acos(
#         math.sin(lat_radians) * math.sin(d_radians)
#         + math.cos(lat_radians) * math.cos(d_radians) * math.cos(SHA_radians)
#     )
#
#     SZA = math.degrees(SZA_radians)
#
#     SEA = 90 - SZA
#
#     if SEA < 0:  # Before Sunrise or after sunset
#         SEA = 0
#
#     return math.radians(SEA)
#
# def fountain_runtime(df, fountain):
#     df["Fountain"] = 0  # Fountain run time
#
#     if option == 'schwarzsee':
#         df["Fountain"] = 0  # Fountain run time
#
#         df_nights = pd.read_csv(
#             os.path.join(folders["dirname"], "data/raw/schwarzsee_fountain_time.txt"),
#             sep="\s+",
#         )
#
#         df_nights["Start"] = pd.to_datetime(
#             df_nights["Date"] + " " + df_nights["start"]
#         )
#         df_nights["End"] = pd.to_datetime(df_nights["Date"] + " " + df_nights["end"])
#         df_nights["Start"] = pd.to_datetime(
#             df_nights["Start"], format="%Y-%m-%d %H:%M:%S"
#         )
#         df_nights["End"] = pd.to_datetime(df_nights["End"], format="%Y-%m-%d %H:%M:%S")
#
#         df_nights["Date"] = pd.to_datetime(df_nights["Date"], format="%Y-%m-%d")
#         mask = (df_nights["Date"] >= dates["start_date"]) & (
#                 df_nights["Date"] <= dates["end_date"]
#         )
#         df_nights = df_nights.loc[mask]
#         df_nights = df_nights.reset_index()
#
#         for i in range(0, df_nights.shape[0]):
#             df_nights.loc[i, "Start"] = df_nights.loc[i, "Start"] - pd.Timedelta(days=1)
#             df.loc[
#                 (df["When"] >= df_nights.loc[i, "Start"])
#                 & (df["When"] <= df_nights.loc[i, "End"]),
#                 "Fountain",
#             ] = 1
#
#     if option == 'temperature':
#         mask = df["T_a"] < fountain["crit_temp"]
#         mask_index = df[mask].index
#         df.loc[mask_index, "Fountain"] = 1
#         mask = df["When"] >= dates["fountain_off_date"]
#         mask_index = df[mask].index
#         df.loc[mask_index, "Fountain"] = 0
#
#
#     if option == "energy":
#
#         """Constants"""
#         Ls = 2848 * 1000  # J/kg Sublimation
#         Le = 2514 * 1000  # J/kg Evaporation
#         Lf = 334 * 1000  # J/kg Fusion
#         cw = 4.186 * 1000  # J/kg Specific heat water
#         ci = 2.108 * 1000  # J/kgC Specific heat ice
#         rho_w = 1000  # Density of water
#         rho_i = 916  # Density of Ice rho_i
#         rho_a = 1.29  # kg/m3 air density at mean sea level
#         k = 0.4  # Van Karman constant
#         bc = 5.670367 * math.pow(10, -8)  # Stefan Boltzman constant
#         g = 9.8  # gravity
#
#         """Miscellaneous"""
#         time_steps = 5 * 60  # s Model time steps
#         p0 = 1013  # Standard air pressure hPa
#
#         """ Estimating Albedo """
#         df["a"] = albedo(df, surface)
#
#         df["T_s"] = 0
#
#         for i in range(1, df.shape[0]):
#
#             """ Energy Balance starts """
#
#             # Vapor Pressure empirical relations
#             if "vp_a" not in list(df.columns):
#                 df.loc[i, "vpa"] = (
#                         6.11
#                         * math.pow(
#                     10, 7.5 * df.loc[i - 1, "T_a"] / (df.loc[i - 1, "T_a"] + 237.3)
#                 )
#                         * df.loc[i, "RH"]
#                         / 100
#                 )
#             else:
#                 df.loc[i, "vpa"] = df.loc[i, "vp_a"]
#
#             df.loc[i, "vp_ice"] = 6.112 * np.exp(
#                 22.46 * (df.loc[i - 1, "T_s"]) / ((df.loc[i - 1, "T_s"]) + 243.12)
#             )
#
#             # Sublimation only
#             df.loc[i, "Ql"] = (
#                     0.623
#                     * Ls
#                     * rho_a
#                     / p0
#                     * math.pow(k, 2)
#                     * df.loc[i, "v_a"]
#                     * (df.loc[i, "vpa"] - df.loc[i, "vp_ice"])
#                     / (
#                             np.log(surface["h_aws"] / surface["z0mi"])
#                             * np.log(surface["h_aws"] / surface["z0hi"])
#                     )
#             )
#
#             # Short Wave Radiation SW
#             df.loc[i, "SW"] = (1 - df.loc[i, "a"]) * (
#                     df.loc[i, "Rad"] + df.loc[i, "DRad"]
#             )
#
#             # Cloudiness from diffuse fraction
#             if df.loc[i, "Rad"] + df.loc[i, "DRad"] > 1:
#                 df.loc[i, "cld"] = df.loc[i, "DRad"] / (
#                         df.loc[i, "Rad"] + df.loc[i, "DRad"]
#                 )
#             else:
#                 df.loc[i, "cld"] = 0
#
#             # atmospheric emissivity
#             df.loc[i, "e_a"] = (
#                                            1.24
#                                            * math.pow(abs(df.loc[i, "vpa"] / (df.loc[i, "T_a"] + 273.15)),
#                                                       1 / 7)
#                                    ) * (1 + 0.22 * math.pow(df.loc[i, "cld"], 2))
#
#             # Long Wave Radiation LW
#             if "oli000z0" not in list(df.columns):
#
#                 df.loc[i, "LW"] = df.loc[i, "e_a"] * bc * math.pow(
#                     df.loc[i, "T_a"] + 273.15, 4
#                 ) - surface["ie"] * bc * math.pow(df.loc[i - 1, "T_s"] + 273.15, 4)
#             else:
#                 df.loc[i, "LW"] = df.loc[i, "oli000z0"] - surface["ie"] * bc * math.pow(
#                     df.loc[i - 1, "T_s"] + 273.15,
#                     4)
#
#             # Sensible Heat Qs
#             df.loc[i, "Qs"] = (
#                     ci
#                     * rho_a
#                     * df.loc[i, "p_a"]
#                     / p0
#                     * math.pow(k, 2)
#                     * df.loc[i, "v_a"]
#                     * (df.loc[i, "T_a"] - df.loc[i - 1, "T_s"])
#                     / (
#                             np.log(surface["h_aws"] / surface["z0mi"])
#                             * np.log(surface["h_aws"] / surface["z0hi"])
#                     )
#             )
#
#             # Total Energy W/m2
#             df.loc[i, "TotalE"] = df.loc[i, "SW"] + df.loc[i, "LW"] + df.loc[i, "Qs"] + df.loc[i, "Ql"]
#
#         x = df["When"]
#         mask = df["TotalE"] < 0
#         mask_index = df[mask].index
#         df.loc[mask_index, "Fountain"] = 1
#         mask = df["When"] >= dates["fountain_off_date"]
#         mask_index = df[mask].index
#         df.loc[mask_index, "Fountain"] = 0
#
#     return df["Fountain"]
#
# def cloudiness(df):
#     S_0 = 1360  # W m-2
#
#     for i in range(1, df.shape[0]) :
#
#         # Vapor Pressure empirical relations
#         if "vp_a" not in list(df.columns):
#             df.loc[i, "vpa"] = (
#                     6.11
#                     * math.pow(
#                 10, 7.5 * df.loc[i - 1, "T_a"] / (df.loc[i - 1, "T_a"] + 237.3)
#             )
#                     * df.loc[i, "RH"]
#                     / 100
#             )
#         else:
#             df.loc[i, "vpa"] = df.loc[i, "vp_a"]
#
#         # Estimating Solar Area fraction
#         df.loc[i, "theta_s"] = getSEA(
#             df.loc[i, "When"],
#             fountain["latitude"],
#             fountain["longitude"],
#             fountain["utc_offset"],
#         )
#
#         optical_air_mass = 35 * math.sin(df.loc[i, "theta_s"] * (math.pow((1224 * math.pow(math.sin(df.loc[i, "theta_s"]), 2) + 1), -0.5)))
#         tau_R_tau_pg = 1.021 - 0.084* math.pow((optical_air_mass * (0.00949 * df.loc[i, "p_a"] + 0.051)), 0.5)
#         precipitable_water = 4650 * df.loc[i, "vpa"] / (df.loc[i, "T_a"] + 273.15)
#         # print(optical_air_mass, precipitable_water)
#         tau_w = 1 - 0.077 * math.pow((optical_air_mass * precipitable_water),0.3)
#         tau_a = math.pow(0.935, optical_air_mass)
#         df.loc[i, "S_clr"] = S_0 * math.sin(df.loc[i, "theta_s"]) * tau_R_tau_pg * tau_w * tau_a
#
#         if df.loc[i, "S_clr"] != 0:
#             df.loc[i, "s"] = (df.loc[i, "Rad"] + df.loc[i, "DRad"]) / df.loc[i, "S_clr"]
#             if df.loc[i, "s"] > 1:
#                 df.loc[i, "s"] = 1
#             df.loc[i, "e_a"] = (1 - df.loc[i, "s"]) + df.loc[i, "s"] * (0.83 - 0.18 * math.pow( 10, -0.067 * df.loc[i, "vpa"]))
#         else:
#             if i - 96 > 0:
#                 for j in range(i - 96, i):
#                     df.loc[i, "e_a"] += df.loc[j, "e_a"]
#             else:
#                 for j in range(i, i + 96):
#                     df.loc[i, "e_a"] += df.loc[j, "e_a"]
#
#                     df.loc[i, "e_a"] = df.loc[i, "e_a"] / 96
#
#         # # Cloudiness from diffuse fraction
#         # if df.loc[i, "Rad"] + df.loc[i, "DRad"] > 1:
#         #     df.loc[i, "cld"] = df.loc[i, "DRad"] / (
#         #             df.loc[i, "Rad"] + df.loc[i, "DRad"]
#         #     )
#         # else:
#         # # Night Cloudiness average of last 8 hours
#         #     if i - 96 > 0:
#         #         for j in range(i - 96, i):
#         #             df.loc[i, "cld"] += df.loc[j, "cld"]
#         #     else:
#         #         for j in range(i, i + 96):
#         #             df.loc[i, "cld"] += df.loc[j, "cld"]
#         #
#         #             df.loc[i, "cld"] = df.loc[i, "cld"] / 96
#
#     x = df.When
#     y = df.S_clr
#     fig, ax = plt.subplots()
#     ax.plot(x, y)
#     plt.show()
#
#     x=df.When
#     y=df.s
#     fig, ax = plt.subplots()
#     ax.plot(x,y)
#     plt.show()
#
#     x = df.When
#     y = df.e_a
#     fig, ax = plt.subplots()
#     ax.plot(x, y)
#     plt.show()
#
#     return df["vpa"], df["theta_s"], df["cld"]

def icestupa(df, fountain, surface):

    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is for temp")
    logger.warning("This is for solid")
    logger.error("This is for melted")
    logger.critical("This is a critical message")

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

    """Miscellaneous"""
    time_steps = 5 * 60  # s Model time steps
    p0 = 1013  # Standard air pressure hPa
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
        "deposition",
    ]
    for col in l:
        df[col] = 0

    """ Estimating Fountain Spray radius """
    R_f = (
        df["r_f"].replace(0, np.NaN).mean()
    )  # todo implement variable spray radius for variable discharge

    """ Simulation """
    for i in range(1, df.shape[0]):

        # Ice Melted
        if df.loc[i - 1, "ice"] <= 0.05:
            if df.Discharge[i:].sum() == 0:  # If ice melted after fountain run
                df.loc[i - 1, "solid"] = 0
                df.loc[i - 1, "ice"] = 0
                df.loc[i - 1, "iceV"] = 0
                break

            else:  # If ice melted in between fountain run
                df.loc[i - 1, "solid"] = 0
                df.loc[i - 1, "ice"] = 0
                df.loc[i - 1, "iceV"] = 0
                state = 0

        # Initiate ice formation
        if (df.loc[i, "Discharge"] > 0) & (state == 0):
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

            logger.info(
                "Ice radius is %s and ice is %s at %s",
                df.loc[i, "r_ice"],
                df.loc[i, "h_ice"],
                df.loc[i, "When"],
            )

            # Initialize AIR ice layer and update
            if ice_layer == 0:

                ice_layer = dx * df.loc[i, "SA"] * rho_i
                logger.warning(
                    "Ice layer initialised %s thick at %s", ice_layer, df.loc[i, "When"]
                )
                df.loc[i - 1, "ice"] = ice_layer

            else:
                ice_layer = dx * df.loc[i, "SA"] * rho_i
                logger.info("Ice layer is %s thick at %s", ice_layer, df.loc[i, "When"])

            # Precipitation to ice quantity
            if df.loc[i, "T_a"] < surface["rain_temp"]:
                df.loc[i, "ppt"] = (
                    surface["snow_fall_density"]
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

                    logger.error(
                        "Ice layer made %s thick ice at %s",
                        df.loc[i, "solid"],
                        df.loc[i, "When"],
                    )
                    df.loc[i, "delta_T_s"] = -df.loc[i - 1, "T_s"]

            """ Energy Balance starts """

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

            if df.loc[i, "Ql"] < 0 :
                df.loc[i, "gas"] -= (df.loc[i, "Ql"] * df.loc[i, "SA"] * time_steps) / Ls

                # Removing gas quantity generated from previous ice
                df.loc[i, "solid"] += (
                    df.loc[i, "Ql"] * (df.loc[i, "SA"]) * time_steps
                ) / Ls

                # Ice Temperature
                df.loc[i, "delta_T_s"] += (
                    df.loc[i, "Ql"] * df.loc[i, "SA"] * time_steps
                ) / (ice_layer * ci)

            else: # Deposition

                df.loc[i, "deposition"] -= (df.loc[i, "Ql"] * df.loc[i, "SA"] * time_steps) / Ls

                # Adding new deposit
                df.loc[i, "solid"] += (
                                              df.loc[i, "Ql"] * (df.loc[i, "SA"]) * time_steps
                                      ) / Ls

            logger.info(
                "Ice made after sublimation is %s thick at %s",
                round(df.loc[i, "solid"]),
                df.loc[i, "When"],
            )


            df.loc[i, "SRf"] = (
                0.5
                * df.loc[i, "h_ice"]
                * df.loc[i, "r_ice"]
                * math.cos(df.loc[i, "theta_s"])
                + math.pi
                * math.pow(df.loc[i, "r_ice"], 2)
                * 0.5
                * math.sin(df.loc[i, "theta_s"])
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

                    logger.warning(
                        "Ice made after energy neg is %s thick at temp %s",
                        round(df.loc[i, "solid"]),
                        df.loc[i - 1, "T_s"],
                    )

                else:
                    """ When fountain off and energy negative """

                    if df.loc[i - 1, "liquid"] < 0:
                        logger.error(
                            "Liquid is %s at %s",
                            round(df.loc[i, "delta_T_s"]),
                            df.loc[i, "When"],
                        )
                        df.loc[i - 1, "liquid"] = 0

                    # Cooling Ice
                    df.loc[i, "delta_T_s"] += (df.loc[i, "EJoules"]) / (ice_layer * ci)

                logger.info(
                    "Ice made after energy neg is %s thick at %s",
                    round(df.loc[i, "solid"]),
                    df.loc[i, "When"],
                )

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

                logger.info(
                    "Ice melted is %s thick at %s",
                    round(df.loc[i, "solid"]),
                    df.loc[i, "When"],
                )

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

            logger.info(
                "Ice volume is %s and meltwater is %s at %s",
                df.loc[i, "ice"],
                df.loc[i, "meltwater"],
                df.loc[i, "When"],
            )

    df = df[start:i]

    print("Ice Mass Remaining", float(df["ice"].tail(1)))
    print("Meltwater", float(df["meltwater"].tail(1)))
    print("Ice Volume Max", float(df["iceV"].max()))
    print("Fountain sprayed", float(df["sprayed"].tail(1)))
    print("Ppt", df["ppt"].sum())
    print("Sublimated", float(df["vapour"].tail(1)))
    print("Model ended", df.loc[i - 1, "When"])
    print("Model runtime", df.loc[i - 1, "When"] - df.loc[start, "When"])
    print("Max growth rate", float(df["solid"].max()/5))
    print(
        "Fountain efficiency",
        float((df["meltwater"].tail(1) + df["ice"].tail(1)) / df["sprayed"].tail(1))
        * 100,
    )

    return df
