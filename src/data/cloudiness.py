from pvlib import location
from pvlib import irradiance
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.dates as mdates
from src.data.config import site, dates, fountain, folders

lat, lon = fountain["latitude"], fountain["longitude"]

df = pd.read_csv(
    folders["input_folder"] + "raw_input.csv", sep=",", header=0, parse_dates=["When"]
)

df["ghi"] = df["SW_direct"] + df["SW_diffuse"]

df.rename(columns={"SW_diffuse": "dif"}, inplace=True)

df = df[["When", "ghi", "dif"]]

df = df.set_index("When").resample("1T").interpolate(method="linear").reset_index()


# Create location object to store lat, lon, timezone
site_location = location.Location(lat, lon)

times = pd.date_range(start=dates["start_date"], end=dates["end_date"], freq="1T")
clearsky = site_location.get_clearsky(times)
# Get solar azimuth and zenith to pass to the transposition function
solar_position = site_location.get_solarposition(times=times, method="ephemeris")
# solar_time = site_location.get_solarposition(times=times, method = 'ephemeris')
solar_df = pd.DataFrame(
    {
        "ghics": clearsky["ghi"],
        "difcs": clearsky["dhi"],
        "zen": solar_position["zenith"],
        "sea": solar_position["elevation"],
    }
)

solar_df.index = solar_df.index.set_names(["When"])
solar_df = solar_df.reset_index()

df = pd.merge(solar_df, df, on="When")

df = df.reset_index()
df["When_UTC"] = df["When"] + pd.DateOffset(hours=1)

df[["ghi", "ghics", "dif", "difcs", "zen", "When_UTC", "sea"]].to_csv(
    folders["input_folder"] + "solar_input.csv", index=False, header=False
)

df1 = pd.read_csv(folders["input_folder"] + "ruizarias.txt", skiprows=5, header=None)

df["cld"] = df1

dfx = df[['When', 'cld']].set_index("When").resample("5T").ffill().reset_index()

print(dfx.head())

dfx.to_csv(
    folders["input_folder"] + "clear_sky.csv"
)

df = df.set_index("When").resample("5T").mean().reset_index()

df["cld"] = dfx["cld"]

df["Dn"] = (df["dif"] - df["difcs"]) / df["ghics"]

for i in range(0, df.shape[0]):
    if df.loc[i, "sea"] < 20:
        df.loc[i, "Dn"] = np.NaN
    if np.isnan(df.loc[i, "Dn"]):
        df.loc[i, "cld"] = np.NaN
    else:
        if df.loc[i, "Dn"] < 0:
            if df.loc[i, "ghi"] / df.loc[i, "ghics"] > 0.4:
                df.loc[i, "Dn"] = 0
            else:
                df.loc[i, "cld"] = 1

        if (df.loc[i, "cld"] == 1) & (df.loc[i, "Dn"] > 0):
            df.loc[i, "cld"] = 2.255 * math.pow(df.loc[i, "Dn"], 0.9381)

df.loc[(df["Dn"] < 0.37) & (df["Dn"] > 0.9), "cld"] = 1
df.loc[(df["cld"] > 1), "cld"] = 1
# df['cld'] = df['cld'].interpolate(method='linear')


# df['cld_11'] = df.loc[:,"cld"].rolling(window=11).mean()
# df['cld'] = df['cld'].fillna(method='bfill')

print(df.head())
df.to_csv(folders["input_folder"] + "solar_output.csv")

pp = PdfPages(folders["input_folder"] + "cloudiness.pdf")
fig, ax1 = plt.subplots()
x = df.When
y1 = df.cld
ax1.scatter(df.When, y1, s=0.1)
ax1.set_ylabel("Cloudiness")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

y1 = df.Dn
ax1 = fig.add_subplot(111)
ax1.scatter(df.When, y1, s=0.1)
ax1.set_ylabel("Dn")
ax1.grid()
ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

fig.autofmt_xdate()
pp.savefig(bbox_inches="tight")
plt.clf()

# y1 = df.cld_11
# ax1 = fig.add_subplot(111)
# ax1.plot(df.When, y1, "k-", linewidth=0.5)
# ax1.set_ylabel("Cloudiness")
# ax1.grid()
# ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
# ax1.xaxis.set_minor_locator(mdates.DayLocator())
#
# fig.autofmt_xdate()
# pp.savefig(bbox_inches="tight")
# plt.clf()

pp.close()
