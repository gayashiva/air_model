from pvlib import location
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


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
    clearsky = site_location.get_clearsky(times=times)

    solar_df = pd.DataFrame(
        {
            "ghics": clearsky["ghi"],
            "dhics": clearsky["dhi"],
            "dnics": clearsky["dni"],
            # "zen": solar_position["zenith"],
            "sea": solar_position["elevation"],
        }
    )
    solar_df.loc[solar_df["sea"] < 0, "sea"] = 0
    solar_df.index = solar_df.index.set_names(["time"])
    solar_df = solar_df.reset_index()
    return solar_df


if __name__ == "__main__":
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 31)
    lat = 34.16779520435076
    long = 77.45920194639129
    alt = 3696
    df = get_solar(lat, long, start, end, 60 * 60, utc=5.5, alt=alt)
    print(df.tail())
    df.to_csv("hial_2020_solar.csv")
