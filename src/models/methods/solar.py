"""Function that returns solar elevation angle
"""
from pvlib import location, irradiance, solarposition
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
    return (today_utc - today_target.tz_convert(tz=utc)).total_seconds() / (60 * 60)

def get_solar(coords, start, end, DT, alt):  
    """
    returns solar angle for each time step
    """

    with open("constants.json") as f:
        CONSTANTS = json.load(f)

    site_location = location.Location(coords[0], coords[1], altitude=alt)

    # utc = get_offset(*coords, date=start)
    utc=-8

    times = pd.date_range(
        start - pd.Timedelta(hours=utc),
        end - pd.Timedelta(hours=utc),
        freq=(str(int(DT / 60)) + "T"),
    )

    solar_position = site_location.get_solarposition(times=times)
    solar_position["R"] = solarposition.nrel_earthsun_distance(time=times)
    earth_semimajor_axis_meters = 149.6 * 10**9
    solar_position["R"] *= earth_semimajor_axis_meters
    # solar_position["hour_angle"] = solarposition.hour_angle(times=times,longitude = coords[1],
    #                                                         equation_of_time=solarposition.equation_of_time_spencer71(dayofyear=times.dayofyear))
    # solar_position["hour_angle"] = np.radians(solar_position["hour_angle"])
    solar_position["declination"] = solarposition.declination_spencer71(dayofyear=times.dayofyear)
    solar_position["hour_angle"] = -np.tan(np.radians(coords[0])) * np.tan(solar_position["declination"].mean())
    solar_position.hour_angle[solar_position["hour_angle"] < -1] = -1
    solar_position.hour_angle[solar_position["hour_angle"] > 1] = 1
    solar_position["hour_angle"] = np.arccos(solar_position["hour_angle"])

    # self.df.loc[i, "tau_atm"] = self.df.loc[i,"SW_global"]/self.df.loc[i,"SW_extra"]
    # self.df.loc[i, "tau_atm"] = clearness_index(ghi = self.df.loc[i,"SW_global"], solar_zenith =
    #                                             solar_position['zenith'], extra_radiation= I0)
    # clearsky = site_location.get_clearsky(times=times, model = 'simplified_solis')
    clearsky = site_location.get_clearsky(times=times, model = 'ineichen')
    # clearness = irradiance.erbs(ghi = clearsky["ghi"], zenith = solar_position['zenith'],
    #                                   datetime_or_doy= times)
    # dni_extra = irradiance.get_extra_radiation(datetime_or_doy= times)

    solar_df = pd.DataFrame(
        {
            "ghi": clearsky["ghi"],
            # "SW_diffuse": clearness["dhi"],
            # "cld": 1 - clearness["kt"],
            "sea": np.radians(solar_position["elevation"]),
            # "SW_extra": irradiance.get_extra_radiation(datetime_or_doy= times)
            # "SW_extra": solar_position["TOA"]
        }
    )
    # solar_df["SW_extra"] = 1366/math.pi * ((earth_semimajor_axis_meters/solar_position["R"])**2) * (solar_position['hour_angle'] * math.sin(coords[0]) *
    #                          math.sin(solar_position['declination']) + math.cos(coords[0]) * math.cos(solar_position['declination'])*math.sin(solar_position['hour_angle']))
    # solar_df["hour_angle"] = np.degrees(solar_position["hour_angle"])
    # solar_df["hour_angle"] = solar_position["hour_angle"]
    # solar_df["R"] = solar_position["R"]
    # solar_df["declination"] = np.degrees(solar_position["declination"])
    # solar_position["hour_angle"] = np.radians(solar_position["hour_angle"])
    # Calculate solar_df["SW_extra"] using element-wise operations
    solar_df["SW_extra"] = (
        # 1366 / (math.pi * solar_position["R"]**2) *
        1366 / math.pi *
        ((earth_semimajor_axis_meters / solar_position["R"])**2) *
        (solar_position['hour_angle'] *
        (np.sin(np.radians(coords[0])) * solar_position['declination'].apply(np.sin) + 
        np.cos(np.radians(coords[0])) * solar_position['declination'].apply(np.cos) * solar_position['hour_angle'].apply(np.sin)))
    )

    bad_values = solar_df["sea"]< 0 
    # solar_df["cld"]= np.where(bad_values, np.nan, solar_df["cld"])

    solar_df["sea"]= np.where(bad_values, 0, solar_df["sea"])
    # cld = round(solar_df["cld"].mean(), 2)
    # solar_df["cld"]= np.where(bad_values, cld, solar_df["cld"])
    # logger.warning("Diffuse and direct SW calculated with cld %s" % cld)

    solar_df.index = solar_df.index.set_names(["time"])
    solar_df = solar_df.reset_index()
    solar_df["time"] += pd.Timedelta(hours=utc)

    return solar_df

if __name__ == "__main__":
    # tf = TimezoneFinder()
    # coords=[46.65549,8.29149]
    # site_location = location.Location(coords[0], coords[1])
    # print(site_location.lookup_altitude(*coords))
    # # coords=[34.216638,77.606949]
    # # print(timezone(tf.certain_timezone_at(lat=coords[0], lng=coords[1])))
    # # print(get_offset(lat=coords[0], lng=coords[1]))
    # print(get_offset(*coords,date=datetime(2021, 12, 3, 8)))

    coords=[49.25,123.1]
    site_location = location.Location(coords[0], coords[1])

    """Solar radiation"""
    solar_df = get_solar(
        coords=coords,
        start=datetime(2023, 6, 21),
        end=datetime(2023, 6, 21, 21),
        DT=3600,
        alt=0,
    )
    print(solar_df[["SW_extra"]])



