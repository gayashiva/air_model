"""Function that returns solar elevation angle
"""
from pvlib import location, irradiance
import numpy as np
import pandas as pd
import logging, json, math
from datetime import datetime
from pytz import timezone, utc
from timezonefinder import TimezoneFinder
from codetiming import Timer

# Module logger
logger = logging.getLogger("__main__")

def get_offset(lat, lng, date):
    """
    returns a location's time zone offset from UTC in minutes.
    """
    tf = TimezoneFinder()
    tz_target = timezone(tf.certain_timezone_at(lng=lng, lat=lat))
    # ATTENTION: tz_target could be None! handle error case
    today_target = tz_target.localize(date)
    today_utc = utc.localize(date) # Note that utc is now 1 for guttannen due to winter time
    return (today_utc - today_target).total_seconds() / (60 * 60)

def get_solar(coords, start, end, DT, alt):  
    """
    returns solar angle for each time step
    """

    with open("data/common/constants.json") as f:
        CONSTANTS = json.load(f)

    site_location = location.Location(coords[0], coords[1], altitude=alt)

    utc = get_offset(*coords, date=start)

    times = pd.date_range(
        start - pd.Timedelta(hours=utc),
        end - pd.Timedelta(hours=utc),
        freq=(str(int(DT / 60)) + "T"),
    )

    solar_position = site_location.get_solarposition(times=times, method="ephemeris")
    clearsky = site_location.get_clearsky(times=times, model = 'simplified_solis')
    # clearsky = site_location.get_clearsky(times=times, model = 'ineichen')
    clearness = irradiance.erbs(ghi = clearsky["ghi"], zenith = solar_position['zenith'],
                                      datetime_or_doy= times) 

    solar_df = pd.DataFrame(
        {
            "ghi": clearsky["ghi"],
            "SW_diffuse": clearness["dhi"],
            "cld": 1 - clearness["kt"],
            "sea": np.radians(solar_position["elevation"]),
        }
    )

    bad_values = solar_df["sea"]< 0 
    solar_df["cld"]= np.where(bad_values, np.nan, solar_df["cld"])

    logger.info(solar_df.cld.describe())
    solar_df["sea"]= np.where(bad_values, 0, solar_df["sea"])
    cld = round(solar_df["cld"].mean(), 2)
    solar_df["cld"]= np.where(bad_values, cld, solar_df["cld"])
    logger.warning("Diffuse and direct SW calculated with cld %s" % cld)

    solar_df.index = solar_df.index.set_names(["time"])
    solar_df = solar_df.reset_index()
    solar_df["time"] += pd.Timedelta(hours=utc)
    # For discharge output

    for i in range(0, solar_df.shape[0]):
        solar_df.loc[i, "f_cone"] = (math.pi * math.sin(solar_df.loc[i, "sea"]) + math.cos(solar_df.loc[i, "sea"]))/(2*math.sqrt(2)*math.pi)

    solar_df["SW_direct"]= solar_df["ghi"] - solar_df["SW_diffuse"]
    solar_df["dis"] = -1 * (1 - CONSTANTS["A_I"]) * (solar_df["SW_direct"] * solar_df["f_cone"] + solar_df["SW_diffuse"]) * CONSTANTS["DT"] / (CONSTANTS["L_F"] * 60)

    return cld, solar_df

if __name__ == "__main__":
    tf = TimezoneFinder()
    coords=[46.65549,8.29149]
    # coords=[34.216638,77.606949]
    # print(timezone(tf.certain_timezone_at(lat=coords[0], lng=coords[1])))
    # print(get_offset(lat=coords[0], lng=coords[1]))
    print(get_offset(*coords,date=datetime(2021, 12, 3, 8)))

