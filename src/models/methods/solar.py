"""Function that returns solar elevation angle
"""
from pvlib import location
import numpy as np
import pandas as pd
import logging
from codetiming import Timer

logger = logging.getLogger(__name__)

# @cache_it(limit=1000, expire=None)
def get_solar(
    latitude, longitude, start, end, DT, utc, alt
):  # Provides solar angle for each time step

    site_location = location.Location(latitude, longitude, tz=utc, altitude=alt)

    times = pd.date_range(
        start,
        end,
        freq=(str(int(DT / 60)) + "T"),
    )

    # Get solar azimuth and zenith to pass to the transposition function
    solar_position = site_location.get_solarposition(times=times, method="ephemeris")
    # clearsky = site_location.get_clearsky(times=times)

    solar_df = pd.DataFrame(
        {
            # "ghics": clearsky["ghi"],
            # "difcs": clearsky["dhi"],
            # "zen": solar_position["zenith"],
            "sea": np.radians(solar_position["elevation"]),
        }
    )
    solar_df.loc[solar_df["sea"] < 0, "sea"] = 0
    solar_df.index = solar_df.index.set_names(["time"])
    solar_df = solar_df.reset_index()
    return solar_df
