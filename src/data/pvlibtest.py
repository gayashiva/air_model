from pvlib import location
from pvlib import irradiance
import pandas as pd
from matplotlib import pyplot as plt
from src.data.config import site, dates, fountain, folders

# For this example, we will be using Golden, Colorado
tz = 'MST'
lat, lon = fountain['latitude'], fountain['longitude']

df = pd.read_csv(folders["input_folder"] + "raw_input.csv" , sep=",", header=0, parse_dates=["When"])

df['ghi'] = df['SW_direct'] + df['SW_diffuse']


df.rename(
            columns={
                "SW_diffuse": 'dif',
            },
            inplace=True,
        )
df = df.set_index("When").resample("1T").interpolate(method='linear').reset_index()

# Create location object to store lat, lon, timezone
site_location = location.Location(lat, lon, tz=tz)

times = pd.date_range(start=dates["start_date"], end=dates["end_date"], freq="1T")
clearsky = site_location.get_clearsky(times)
# Get solar azimuth and zenith to pass to the transposition function
solar_position = site_location.get_solarposition(times=times, method = 'ephemeris')
# solar_time = site_location.get_solarposition(times=times, method = 'ephemeris')
solar_df = pd.DataFrame({'ghics': clearsky['ghi'],'difcs': clearsky['dhi'],
                         'zen': solar_position['zenith'], 'solar_time': solar_position['solar_time'],
                         })



solar_df.index = solar_df.index.set_names(['When'])
solar_df = solar_df.reset_index()

df = pd.merge(
    solar_df,
    df,
    on="When",
)

df = df.reset_index()
df["When"] = df["When"] + pd.DateOffset(hours=1)
# df["When"] = df["When"].dt.strftime("%m/%d/%y")
df["solar_time"] = df["solar_time"].round(2)


df["solar_time"] = df["When"].dt.strftime("%m/%d/%y").apply(str) + " " + df["solar_time"].apply(str)
print(df[["When",'solar_time']])

df = df[["ghi", "ghics", "dif", "difcs", 'zen', "When"]]
# df.loc[df['zen']>90, 'zen'] = 0
print(df.head(30))
df.to_csv(folders["input_folder"] + "solar_input.csv", index=False, header=False)


# # Get irradiance data for summer and winter solstice, assuming 25 degree tilt
# # and a south facing array
# summer_irradiance = get_irradiance(site, '06-20-2020', 25, 180)
# winter_irradiance = get_irradiance(site, '12-21-2020', 25, 180)
#
# # Convert Dataframe Indexes to Hour:Minute format to make plotting easier
# summer_irradiance.index = summer_irradiance.index.strftime("%H:%M")
# winter_irradiance.index = winter_irradiance.index.strftime("%H:%M")
#
# # Plot GHI vs. POA for winter and summer
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# summer_irradiance['GHI'].plot(ax=ax1, label='GHI')
# summer_irradiance['POA'].plot(ax=ax1, label='POA')
# winter_irradiance['GHI'].plot(ax=ax2, label='GHI')
# winter_irradiance['POA'].plot(ax=ax2, label='POA')
# ax1.set_xlabel('Time of day (Summer)')
# ax2.set_xlabel('Time of day (Winter)')
# ax1.set_ylabel('Irradiance ($W/m^2$)')
# ax1.legend()
# ax2.legend()
# plt.show()