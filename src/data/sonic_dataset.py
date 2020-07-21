import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
from tqdm import tqdm
import os
import glob
from src.data.config import site, dates, folders, fountain, surface
import fnmatch


dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

start = time.time()

if __name__ == '__main__':

    path = folders['data']  # use your path
    all_files = glob.glob(
        os.path.join(path, "TOA5__Flux_CSF*.dat"))
    all_files_B = glob.glob(
        os.path.join(path, "TOA5__FluxB_CSF*.dat"))
    li = []
    li_B = []
    for filename in all_files:
        df_in = pd.read_csv(filename, header=1)
        df_in = df_in[2:].reset_index(drop=True)
        li.append(df_in)

    for filename in all_files_B:
            df_inB = pd.read_csv(filename, header=1)
            df_inB = df_inB[2:].reset_index(drop=True)
            df_inB = df_inB.drop(["RECORD"], axis=1)
            li_B.append(df_inB)

    df_A = pd.concat(li, axis=0, ignore_index=True)
    df_B = pd.concat(li_B, axis=0, ignore_index=True)

    for col in df_B.columns:
        if 'B' not in col:
            if col != 'TIMESTAMP':
                df_B = df_B.drop([col], axis=1)


    df_A["TIMESTAMP"] = pd.to_datetime(df_A["TIMESTAMP"], format='%Y-%m-%d %H:%M:%S')
    df_B["TIMESTAMP"] = pd.to_datetime(df_B["TIMESTAMP"], format='%Y-%m-%d %H:%M:%S')

    df_A = df_A.sort_values(by='TIMESTAMP')
    df_B = df_B.sort_values(by='TIMESTAMP')

    df = pd.merge(df_A, df_B, how='inner', left_index=True, on='TIMESTAMP')

    cols = df.columns.drop('TIMESTAMP')

    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
            
    # Errors
    df["SnowHeight"] = df["SnowHeight"] - 10
    df['H'] = df['H'] / 1000
    df['HB'] = df['HB'] / 1000
    # df['H'] = df['H'].apply(lambda x: x if abs(x) < 1000 else np.NAN)
    # df['HB'] = df['HB'].apply(lambda x: x if abs(x) < 1000 else np.NAN)

    df.to_csv(folders["input_folder"] + "raw_output.csv")

    mask = (df["TIMESTAMP"] >= dates["start_date"]) & (df["TIMESTAMP"] <= dates["end_date"])
    df = df.loc[mask]
    df = df.reset_index()

    print(df.info())

    # df = df.fillna(method='ffill')


    df['day'] = df.SW_IN > 10
    df['night'] = df.SW_IN <= 10
    dfday = df[df.day].set_index("TIMESTAMP").resample("D").mean().reset_index()
    dfnight = df[df.night].set_index("TIMESTAMP").resample("D").mean().reset_index()  

    dfd = df.set_index("TIMESTAMP").resample("D").mean().reset_index()

    print(df['H'].corr(df['HB']))

    """Input Plots"""

    pp = PdfPages(folders["input_folder"] + site + "_data" + ".pdf")

    x = df.TIMESTAMP

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    y1 = df.T_probe_Avg
    ax1.plot(x, y1, "k-", linewidth=0.5)
    ax1.set_ylabel("Temperature [$\\degree C$]")
    ax1.grid()

    # Add labels to the plot
    style = dict(size=10, color='gray', rotation=90)

    ax1.text(datetime(2020, 2, 17, 13,35), 0, "Sonic Lower(A) failed", **style)
    ax1.text(datetime(2020, 2, 16, 2,34), 0, "Sonic Upper(B) failed", **style)

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    # ax1 = fig.add_subplot(111)

    # y2 = df.WaterFlow
    # ax1.plot(x, y2, "k-", linewidth=0.5)
    # ax1.set_ylabel("Discharge Rate ")
    # ax1.grid()

    # ax1t = ax1.twinx()
    # ax1t.plot(x, df.Waterpressure, "b-", linewidth=0.5)
    # ax1t.set_ylabel("Water Pressure [$Bar$]", color="b")
    # for tl in ax1t.get_yticklabels():
    #     tl.set_color("b")

    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.HourLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    y1 = df.RH_probe_Avg
    ax1.plot(x, y1, "k-", linewidth=0.5)
    ax1.set_ylabel("Relative Humidity [$\\%$]")
    ax1.grid()

    ax1t = ax1.twinx()
    ax1t.plot(x, df.SnowHeight, "b-", linewidth=0.5)
    ax1t.set_ylabel("Snow Height ", color="b")
    for tl in ax1t.get_yticklabels():
        tl.set_color("b")

    # Add labels to the plot
    style = dict(size=10, color='gray', rotation=90)

    ax1.text(datetime(2020, 2, 17, 13,35), 50, "Sonic Lower(A) failed", **style)
    ax1.text(datetime(2020, 2, 16, 2,34), 50, "Sonic Upper(B) failed", **style)

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    # ax1 = fig.add_subplot(111)

    # y3 = df.NETRAD
    # ax1.plot(x, y3, "k-", linewidth=0.5)
    # ax1.set_ylabel("Net Radiation [$W\\,m^{-2}$]")
    # ax1.grid()

    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.HourLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    ax1 = fig.add_subplot(111)
    y31 = -df.H
    ax1.plot(x, y31, "k-", linewidth=0.5)
    ax1.set_ylabel("Sensible Heat A [$W\\,m^{-2}$]")
    ax1.grid()

    ax1t = ax1.twinx()
    ax1t.plot(x, df.H_QC, "b-", linewidth=0.5)
    ax1t.set_ylabel("Sensible Heat A Quality ", color="b")
    for tl in ax1t.get_yticklabels():
        tl.set_color("b")

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    # ax1 = fig.add_subplot(111)
    # y3 = df.H_QC
    # ax1.plot(x, y3, "k-", linewidth=0.5)
    # ax1.set_ylabel("Sensible Heat A Quality")
    # ax1.grid()

    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)
    # y3 = df.TAU_QC
    # ax1.plot(x, y3, "k-", linewidth=0.5)
    # ax1.set_ylabel("Temperature A Quality")
    # ax1.grid()

    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)
    # ax1.scatter(dfday.H, dfnight.H, s=2)
    # ax1.set_xlabel("Day Sensible Heat A[$W\\,m^{-2}$]")
    # ax1.set_ylabel("Night Sensible Heat A [$W\\,m^{-2}$]")
    # ax1.grid()


    # lims = [
    # np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    # np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    # ]

    # # now plot both limits against eachother
    # ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
    # ax1.set_aspect('equal')
    # ax1.set_xlim(lims)
    # ax1.set_ylim(lims)
    # # format the ticks

    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    ax1 = fig.add_subplot(111)

    y31 = -df.HB
    ax1.plot(x, y31, "k-", linewidth=0.5)
    ax1.set_ylabel("Sensible Heat B [$W\\,m^{-2}$]")
    ax1.grid()

    ax1t = ax1.twinx()
    ax1t.plot(x, df.H_QCB, "b-", linewidth=0.5)
    ax1t.set_ylabel("Sensible Heat B Quality ", color="b")
    for tl in ax1t.get_yticklabels():
        tl.set_color("b")

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    # ax1 = fig.add_subplot(111)
    # ax1.scatter(dfday.HB, dfnight.HB, s=2)
    # ax1.set_xlabel("Day Sensible Heat B [$W\\,m^{-2}$]")
    # ax1.set_ylabel("Night Sensible Heat B [$W\\,m^{-2}$]")
    # ax1.grid()


    # lims = [
    # np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    # np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    # ]

    # # now plot both limits against eachother
    # ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
    # ax1.set_aspect('equal')
    # ax1.set_xlim(lims)
    # ax1.set_ylim(lims)
    # # format the ticks

    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)

    # for i in range(1,3):
    #     col = 'Tice_Avg(' + str(i) + ')'
    #     plt.plot(x, df[col], label='id %s' % i)
    # plt.legend()
    
    # ax1.set_ylabel("Ice Temperatures")
    # ax1.set_ylim([-1,3])
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)

    # y4 = df.SnowHeight
    # ax1.plot(x, y4, "k-", linewidth=0.5)
    # ax1.set_ylabel("Snow Height [$cm$]")
    # ax1.grid()

    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.HourLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    # ax1 = fig.add_subplot(111)

    # y5 = df.amb_press_Avg
    # ax1.plot(x, y5, "k-", linewidth=0.5)
    # ax1.set_ylabel("Pressure [$hPa$]")
    # ax1.grid()

    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.DayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.HourLocator())
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    ax1 = fig.add_subplot(111)

    y6 = df.WS
    ax1.plot(x, y6, "k-", linewidth=0.5)
    ax1.set_ylabel("Wind A [$m\\,s^{-1}$]")
    ax1.grid()

    ax1t = ax1.twinx()
    ax1t.plot(x, df.H_QC, "b-", linewidth=0.5)
    ax1t.set_ylabel("Sensible Heat A Quality ", color="b")
    for tl in ax1t.get_yticklabels():
        tl.set_color("b")

    ax1.text(datetime(2020, 2, 17, 13,35), 0, "Sonic Lower(A) failed", **style)
    ax1.text(datetime(2020, 2, 16, 2,34), 0, "Sonic Upper(B) failed", **style)

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    ax1 = fig.add_subplot(111)
    y7 = df.WSB
    ax1.plot(x, y7, "k-", linewidth=0.5)
    ax1.set_ylabel("Wind B [$m\\,s^{-1}$]")
    ax1.grid()

    ax1t = ax1.twinx()
    ax1t.plot(x, df.H_QCB, "b-", linewidth=0.5)
    ax1t.set_ylabel("Sensible Heat B Quality ", color="b")
    for tl in ax1t.get_yticklabels():
        tl.set_color("b")

    ax1.text(datetime(2020, 2, 17, 13,35), 0, "Sonic Lower(A) failed", **style)
    ax1.text(datetime(2020, 2, 16, 2,34), 0, "Sonic Upper(B) failed", **style)

    # format the ticks
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.HourLocator())
    fig.autofmt_xdate()
    pp.savefig(bbox_inches="tight")
    plt.clf()

    # ax1 = fig.add_subplot(111)
    # ax1.scatter(dfday.H, dfday.HB, s=2, color ="blue", label = "Day")
    # ax1.scatter(dfnight.H, dfnight.HB, s=2, color="orange", label = "Night")

    # ax1.set_xlabel("Sonic A Sensible Heat [$W\\,m^{-2}$]")
    # ax1.set_ylabel("Sonic B Sensible Heat [$W\\,m^{-2}$]")
    # ax1.grid()
    # ax1.legend()


    # lims = [
    # np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    # np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
    # ]
    # lims = [
    # np.min([-100, 100]),  # min of both axes
    # np.max([-100, 100]),  # max of both axes
    # ]

    # # now plot both limits against eachother
    # ax1.plot(lims, lims, '--k', alpha=0.25, zorder=0)
    # ax1.set_aspect('equal')
    # ax1.set_xlim(lims)
    # ax1.set_ylim(lims)
    
    # pp.savefig(bbox_inches="tight")
    # plt.clf()

    pp.close()

    # # CSV output
    # df.rename(
    #     columns={
    #         "TIMESTAMP": "When",
    #         "LW_IN": "oli000z0",
    #         "SW_IN": "Rad",
    #         "amb_press_Avg": "p_a",
    #         "WSB": "v_a",
    #         "e_probe": "vp_a",
    #         "WaterFlow": "Discharge",
    #         "T_probe_Avg": "T_a",
    #         "RH_probe_Avg": "RH",
    #         "HB": "H_eddy",
    #     },
    #     inplace=True,
    # )
    
    # for i in tqdm(range(1, df.shape[0])):
    
    #     if np.isnan(df.loc[i, "Discharge"]) or np.isinf(df.loc[i, "Discharge"]):
    #         df.loc[i, "Discharge"] = df.loc[i - 1, "Discharge"]
    #     if np.isnan(df.loc[i, "H_eddy"]) or np.isinf(df.loc[i, "H_eddy"]):
    #         df.loc[i, "H_eddy"] = df.loc[i - 1, "H_eddy"]
    
    #     """Solar Elevation Angle"""
    #     df.loc[i, "SEA"] = getSEA(
    #         df.loc[i, "When"],
    #         fountain["latitude"],
    #         fountain["longitude"],
    #         fountain["utc_offset"],
    #     )
    
    # df = df.set_index("When").resample("30T").ffill().reset_index()
    
    # df_out = df[
    #     ["When", "T_a", "RH", "v_a", "Discharge", "Rad", "DRad", "Prec", "p_a", "SEA", "vp_a", "H_eddy"]
    # ]
    
    # print(df_out.head())
    
    # df_out = df_out.round(5)
    
    
    # filename = folders["input_folder"] + site
    
    # df_out.to_csv(filename + "_raw_input.csv")
    
    # """ Derived Parameters"""
    
    # l = [
    #     "a",
    #     "r_f",
    # ]
    # for col in l:
    #     df[col] = 0
    
    # """Albedo Decay"""
    # surface["decay_t"] = (
    #         surface["decay_t"] * 24 * 60 / 30
    # )  # convert to 30 minute time steps
    # s = 0
    # f = 0
    
    # for i in tqdm(range(1, df.shape[0])):
    
    #     if option == "schwarzsee":
    
    #         ti = surface["decay_t"]
    #         a_min = surface["a_i"]
    
    #         # Precipitation
    #         if (df.loc[i, "Discharge"] == 0) & (df.loc[i, "Prec"] > 0):
    #             if df.loc[i, "T_a"] < surface["rain_temp"]:  # Snow
    #                 s = 0
    #                 f = 0
    
    #         if df.loc[i, "Discharge"] > 0:
    #             f = 1
    #             s = 0
    
    #         if f == 0:  # last snowed
    #             df.loc[i, "a"] = a_min + (surface["a_s"] - a_min) * math.exp(-s / ti)
    #             s = s + 1
    #         else:  # last sprayed
    #             df.loc[i, "a"] = a_min
    #             s = s + 1
    #     else:
    #         df.loc[i, "a"] = surface["a_i"]
    
    #     """ Fountain Spray radius """
    #     if df.loc[i, 'When'] > datetime(2020, 1, 9):
    #         fountain["aperture_f"] = 0.003
    #     else:
    #         fountain["aperture_f"] = 0.005
    
    #     Area = math.pi * math.pow(fountain["aperture_f"], 2) / 4
    #     v_f = df.loc[i, "Discharge"] / (60 * 1000 * Area)
    #     df.loc[i, "r_f"] = projectile_xy(
    #         v_f, fountain["h_f"]
    #     )
    
    # df.to_csv(filename + "_input.csv")

    # pp = PdfPages(folders["input_folder"] + site + "_derived_parameters" + ".pdf")
    
    # x = df.When
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    
    # y1 = df.Discharge
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Discharge [$l\\, min^{-1}$]")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    
    # y1 = df.r_f
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Spray Radius [$m$]")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df.vp_a
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Vapour Pressure")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df.a
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Albedo")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # pp.close()

    # pp = PdfPages(folders["input_folder"] + "schwarzsee_2020.pdf")
    
    # x = df.When
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    
    # y1 = df.WaterFlow
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Discharge [$l\\, min^{-1}$]")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    
    # y1 = df.T_probe_Avg
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Temperature")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    
    # y1 = df.RH_probe_Avg
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Humidity")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df.WS
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Wind speed")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df.SnowHeight
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("SnowHeight")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # # y1 = df_in["Tice"]
    # for i in range(1,3):
    #     col = 'Tice_Avg(' + str(i) + ')'
    #     plt.plot(x, df[col], label='id %s' % i)
    # plt.legend()
    
    # ax1.set_ylabel("Ice Temperatures")
    # ax1.set_ylim([-1,3])
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df.NETRAD
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("NETRAD")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df.H
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Sensible heat")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # y1 = df['H'] + df['NETRAD']
    # ax1.plot(x, y1, "k-", linewidth=0.5)
    # ax1.set_ylabel("Sensible heat")
    # ax1.grid()
    
    # # format the ticks
    # ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    # ax1.xaxis.set_minor_locator(mdates.DayLocator())
    # ax1.grid()
    # fig.autofmt_xdate()
    # pp.savefig(bbox_inches="tight")
    # plt.clf()
    
    # pp.close()