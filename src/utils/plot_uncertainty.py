import matplotlib.pyplot as plt
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
import numpy as np
import pandas as pd
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import uncertainpy as un
import statistics as st
from datetime import datetime, timedelta
from src.utils.settings import config
from src.models.icestupaClass import Icestupa
import matplotlib.dates as mdates

if __name__ == "__main__":
    answers = dict(
        location="Guttannen 2021",
    )

    # Get settings for given location and trigger
    SITE, FOLDER = config(answers["location"])
    input = FOLDER["sim"] + "/"
    output = FOLDER["sim"] + "/"

    names = "full"
    variance = []
    mean = []
    evaluations = []

    data = un.Data()
    filename1 = input + names + ".h5"
    data.load(filename1)
    print(data)

    eval = data["max_volume"].evaluations
    print(
        f"95 percent confidence interval caused by {names} is {round(2 * st.stdev(eval),2)}"
    )

    print(data["max_volume"].mean)

    icestupa = Icestupa("guttannen21")
    icestupa.read_output()
    icestupa.self_attributes()
    df = icestupa.df

    days = pd.date_range(
        start=SITE["start_date"],
        end=SITE["start_date"]+ timedelta(hours=180 * 24 - 1),
        freq="1H",
    )

    data = data["guttannen21"]
    # print(len(data.percentile_5))

    data["When"] = days

    # data['When'] = data['When'] +  pd.to_timedelta(data.time, unit='D')

    # data["IceV"] = df["IceV"]

    blue = "#0a4a97"
    red = "#e23028"
    purple = "#9673b9"
    green = "#28a745"
    orange = "#ffc107"
    pink = "#ce507a"
    skyblue = "#9bc4f0"
    grey = '#ced4da'

    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"

    df_c = pd.read_hdf(FOLDER["input"] + "model_input_" + icestupa.trigger + ".h5", "df_c")
    df_c = df_c[["When", "DroneV", "DroneVError"]]
    if icestupa.name in ["guttannen21", "guttannen20"]:
        df_c = df_c[1:]
    tol = pd.Timedelta('15T')
    df_c = df_c.set_index("When")
    icestupa.df= icestupa.df.set_index("When")
    df_c = pd.merge_asof(left=icestupa.df,right=df_c,right_index=True,left_index=True,direction='nearest',tolerance=tol)
    df_c = df_c[["DroneV","DroneVError", "iceV"]]
    icestupa.df= icestupa.df.reset_index()

    fig, ax = plt.subplots()
    x = df.When
    y1 = df.iceV
    y2 = df_c.DroneV
    ax.set_ylabel("Ice Volume[$m^3$]")
    ax.fill_between(
        data["When"],
        data.percentile_5,
        data.percentile_95,
        color="skyblue",
        alpha=0.3,
        label="90% prediction interval",
    )
    ax.fill_between(data["When"], y1=icestupa.V_dome, y2=0, color=grey, label = "Dome Volume")
    ax.scatter(x, y2, color=CB91_Green, label="Measured Volume")
    ax.plot(x, y1, "b-", label="Modelled Ice Volume", linewidth=1, color=CB91_Blue)
    ax.set_ylim(bottom=0)
    plt.legend()
    # plt.legend(["Mean", "90% prediction interval", "Std", "Validation Measurement"], loc="best")

    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(FOLDER["sim"]+ "uncertainty.jpg", bbox_inches="tight", dpi=300)
    # plt.savefig(FOLDERS["output_folder"] + "jpg/Figure_8.jpg", dpi=300, bbox_inches="tight")
    # plt.show()
