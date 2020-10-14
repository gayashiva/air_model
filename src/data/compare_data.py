import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import register_matplotlib_converters
import os
from sklearn.metrics import mean_squared_error


from src.data.config import site, dates, folders

df = pd.read_csv(folders["input_folder"] + "raw_input.csv", sep=",", header=0, parse_dates=["When"])
df_in3 = pd.read_csv(folders["raw_folder"]+ site + "_ERA5.csv", sep=",", header=0, parse_dates=["dataDate"])

df_in3 = df_in3.drop(["Latitude", "Longitude"], axis=1)
df_in3["time"] = df_in3["validityTime"].replace([0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
                                        ["0000", "0100", "0200", "0300", "0400", "0500", "0600", "0700", "0800",
                                         "0900"])
df_in3["time"] = df_in3["time"].astype(str)
df_in3["time"] = df_in3["time"].str[0:2] + ":" + df_in3["time"].str[2:4]
df_in3["dataDate"] = df_in3["dataDate"].astype(str)
df_in3["When"] = df_in3["dataDate"] + " " + df_in3["time"]
df_in3["When"] = pd.to_datetime(df_in3["When"])

days = pd.date_range(start=dates["start_date"], end=df_in3["When"].iloc[-1], freq="1H")

# print(df5.columns)
# df5["Wind Speed"] = df5["v_a"]
# df["Wind Speed"] = df["v_a"]
# df["Temperature"] = df["T_a"]
# df5["Temperature"] = df5["T_a"]


# dfp = pd.read_csv(
#     os.path.join(folders["raw_folder"], "plaffeien_aws.txt"),
#     sep=";",
#     skiprows=2,
# )
# dfp["When"] = pd.to_datetime(dfp["time"], format="%Y%m%d%H%M")
# dfp["Humidity"] = pd.to_numeric(dfp["ure200s0"], errors="coerce")
# dfp["Wind Speed"] = pd.to_numeric(dfp["fkl010z0"], errors="coerce")
# dfp["Temperature"] = pd.to_numeric(dfp["tre200s0"], errors="coerce")
# dfp["Pressure"] = pd.to_numeric(dfp["prestas0"], errors="coerce")
# dfp["Prec"] = pd.to_numeric(dfp["rre150z0"], errors="coerce")
# dfp["Prec"] = dfp["Prec"] / (2*1000)  # 5 minute sum
# dfp = dfp.set_index("When").resample("5T").interpolate(method='linear').reset_index()

# mask = (dfp["When"] >= dates["start_date"]) & (dfp["When"] <= dates["end_date"])
# dfp = dfp.loc[mask]
# dfp = dfp.reset_index()
#
# dfp = dfp.set_index("When").resample("5T").interpolate(method='linear').reset_index()
#
# dfp = dfp[["When", "Wind Speed", "Temperature", "Prec"]]
# df5 = df5[["When", "Wind Speed", "Temperature", "Prec"]]
# df = df[["When", "Wind Speed", "Temperature", "Prec"]]
#
# print(df5.head())
# print(dfp.head())
#
# print(df5["Prec"].max(), dfp["Prec"].max())
#
# pp = PdfPages(folders["input_folder"] + "compare" + ".pdf")
# fig = plt.figure()
#
# ax1 = fig.add_subplot(111)
# ax1.scatter(df.Temperature, dfp.Temperature, s=2)
# ax1.set_xlabel("Schwarzsee Temp")
# ax1.set_ylabel("Plaffeien Temp")
# ax1.grid()
#
#
# lims = [
# np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
# np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
# ]
#
# # now plot both limits against eachother
# ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
# ax1.set_aspect('equal')
# ax1.set_xlim(lims)
# ax1.set_ylim(lims)
# # format the ticks
#
# pp.savefig(bbox_inches="tight")
# plt.clf()
#
# ax1 = fig.add_subplot(111)
# ax1.scatter(df.Temperature, df5.Temperature, s=2)
# ax1.set_xlabel("Schwarzsee Temp")
# ax1.set_ylabel("ERA5 Temp")
# ax1.grid()
#
#
# lims = [
# np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
# np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
# ]
#
# # now plot both limits against eachother
# ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
# ax1.set_aspect('equal')
# ax1.set_xlim(lims)
# ax1.set_ylim(lims)
# # format the ticks
#
# pp.savefig(bbox_inches="tight")
# plt.clf()
#
# ax1 = fig.add_subplot(111)
# ax1.scatter(df["Wind Speed"], dfp["Wind Speed"], s=2)
# ax1.set_xlabel("Schwarzsee Wind Speed")
# ax1.set_ylabel("Plaffeien Wind Speed")
# ax1.grid()
#
#
# lims = [
# np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
# np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
# ]
#
# # now plot both limits against eachother
# ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
# ax1.set_aspect('equal')
# ax1.set_xlim(lims)
# ax1.set_ylim(lims)
# # format the ticks
#
# pp.savefig(bbox_inches="tight")
# plt.clf()
#
# ax1 = fig.add_subplot(111)
# ax1.scatter(df["Wind Speed"], df5["Wind Speed"], s=2)
# ax1.set_xlabel("Schwarzsee Wind Speed")
# ax1.set_ylabel("ERA5 Wind Speed")
# ax1.grid()
#
#
# lims = [
# np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
# np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
# ]
#
# # now plot both limits against eachother
# ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
# ax1.set_aspect('equal')
# ax1.set_xlim(lims)
# ax1.set_ylim(lims)
# # format the ticks
#
# pp.savefig(bbox_inches="tight")
# plt.clf()
#
# ax1 = fig.add_subplot(111)
# ax1.scatter(df5["Prec"], dfp["Prec"], s=2)
# ax1.set_xlabel("ERA5 Prec")
# ax1.set_ylabel("Plaffeien Prec")
# ax1.grid()
#
#
# lims = [
# np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
# np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
# ]
#
# # now plot both limits against eachother
# ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
# ax1.set_aspect('equal')
# ax1.set_xlim(lims)
# ax1.set_ylim(lims)
# # format the ticks
#
# pp.savefig(bbox_inches="tight")
# plt.clf()
#
# pp.close()