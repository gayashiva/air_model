import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

def summary_figures(self):

    # # Class variables used
    # df = self.df
    # trigger = self.trigger
    # RHO_I = self.RHO_I
    # TIME_STEP = self.TIME_STEP

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    output = self.output
    CB91_Blue = "#2CBDFE"
    CB91_Green = "#47DBCD"
    CB91_Pink = "#F3A0F2"
    CB91_Purple = "#9D2EC5"
    CB91_Violet = "#661D98"
    CB91_Amber = "#F5B14C"
    # grey = '#ced4da'
    self.df = self.df.rename(
        {
            "SW": "$q_{SW}$",
            "LW": "$q_{LW}$",
            "Qs": "$q_S$",
            "Ql": "$q_L$",
            "Qf": "$q_{F}$",
            "Qg": "$q_{G}$",
        },
        axis=1,
    )

    mask = self.df.missing != 1
    nmask = self.df.missing != 0
    ymask = self.df.missing == 1
    pmask = self.df.missing != 2
    df_ERA5 = self.df.copy()
    df_ERA52 = self.df.copy()
    df_SZ = self.df.copy()
    df_SZ.loc[
        ymask, ["When", "T_a", "SW_direct", "SW_diffuse", "v_a", "p_a", "RH"]
    ] = np.NaN
    df_ERA5.loc[
        mask, ["When", "T_a", "SW_direct", "SW_diffuse", "v_a", "p_a", "RH"]
    ] = np.NaN
    df_ERA52.loc[
        pmask, ["When", "T_a", "SW_direct", "SW_diffuse", "v_a", "p_a", "RH"]
    ] = np.NaN

    events = np.split(df_ERA5.When, np.where(np.isnan(df_ERA5.When.values))[0])
    # removing NaN entries
    events = [
        ev[~np.isnan(ev.values)] for ev in events if not isinstance(ev, np.ndarray)
    ]
    # removing empty DataFrames
    events = [ev for ev in events if not ev.empty]
    events2 = np.split(df_ERA52.When, np.where(np.isnan(df_ERA52.When.values))[0])
    # removing NaN entries
    events2 = [
        ev[~np.isnan(ev.values)] for ev in events2 if not isinstance(ev, np.ndarray)
    ]
    # removing empty DataFrames
    events2 = [ev for ev in events2 if not ev.empty]

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
        nrows=6, ncols=1, sharex="col", sharey="row", figsize=(12, 18)
    )

    x = self.df.When

    y1 = self.df.Discharge
    ax1.plot(x, y1, linestyle="-", color="#284D58", linewidth=1)
    ax1.set_ylabel("Fountain Spray [$l\\, min^{-1}$]")

    ax1t = ax1.twinx()
    ax1t.plot(
        x,
        self.df.Prec,
        linestyle="-",
        color=CB91_Blue,
        label="Plaffeien",
    )
    ax1t.set_ylabel("Precipitation [$mm$]", color=CB91_Blue)
    for tl in ax1t.get_yticklabels():
        tl.set_color(CB91_Blue)

    y2 = df_SZ.T_a
    y2_ERA5 = df_ERA5.T_a
    ax2.plot(x, y2, linestyle="-", color="#284D58", linewidth=1)
    for ev in events:
        ax2.axvspan(
            ev.head(1).values, ev.tail(1).values, facecolor="xkcd:grey", alpha=0.25
        )
    ax2.plot(x, y2_ERA5, linestyle="-", color="#284D58")
    ax2.set_ylabel("Temperature [$\\degree C$]")

    y3 = self.df.SW_direct
    lns2 = ax3.plot(
        x, y3, linestyle="-", label="Shortwave Direct", color=CB91_Amber
    )
    lns1 = ax3.plot(
        x,
        self.df.SW_diffuse,
        linestyle="-",
        label="Shortwave Diffuse",
        color=CB91_Blue,
    )
    lns3 = ax3.plot(
        x, self.df.LW_in, linestyle="-", label="Longwave", color=CB91_Green
    )
    ax3.axvspan(
        self.df.When.head(1).values,
        self.df.When.tail(1).values,
        facecolor="grey",
        alpha=0.25,
    )
    ax3.set_ylabel("Radiation [$W\\,m^{-2}$]")

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, ncol=3, loc="best")

    y4 = df_SZ.RH
    y4_ERA5 = df_ERA5.RH
    ax4.plot(x, y4, linestyle="-", color="#284D58", linewidth=1)
    ax4.plot(x, y4_ERA5, linestyle="-", color="#284D58")
    for ev in events:
        ax4.axvspan(
            ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
        )
    ax4.set_ylabel("Humidity [$\\%$]")

    y5 = df_SZ.p_a
    y5_ERA5 = df_ERA5.p_a
    ax5.plot(x, y5, linestyle="-", color="#264653", linewidth=1)
    ax5.plot(x, y5_ERA5, linestyle="-", color="#284D58")
    for ev in events:
        ax5.axvspan(
            ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
        )
    ax5.set_ylabel("Pressure [$hPa$]")

    y6 = df_SZ.v_a
    y6_ERA5 = df_ERA5.v_a
    y6_ERA52 = df_ERA52.v_a
    ax6.plot(x, y6, linestyle="-", color="#264653", linewidth=1, label="Schwarzsee")
    ax6.plot(x, y6_ERA5, linestyle="-", color="#284D58")
    ax6.plot(x, y6_ERA52, linestyle="-", color="#284D58")
    for ev in events: # Creates DeprecationWarning
        ax6.axvspan(
            ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
        )
    for ev in events2:
        ax6.axvspan(
            ev.head(1).values, ev.tail(1).values, facecolor="grey", alpha=0.25
        )
    ax6.set_ylabel("Wind speed [$m\\,s^{-1}$]")

    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        output+ "paper_figures/Model_Input_" + self.trigger + ".jpg", dpi=300, bbox_inches="tight"
    )
    plt.clf()

    fig = plt.figure(figsize=(12, 14))
    dfds = self.df[
        [
            "When",
            "solid",
            "ppt",
            "dpt",
            "cdt",
            "melted",
            "gas",
            "SA",
            "iceV",
            "Discharge",
        ]
    ]

    with pd.option_context("mode.chained_assignment", None):
        for i in range(1, dfds.shape[0]):
            if self.df.loc[i, "SA"] != 0:
                dfds.loc[i, "solid"] = dfds.loc[i, "solid"] / (
                    self.df.loc[i, "SA"] * self.RHO_I
                )
                dfds["solid"] = dfds.loc[dfds.solid >= 0, "solid"]
                dfds.loc[i, "melted"] *= -1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "gas"] *= -1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "ppt"] *= 1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "dpt"] *= 1 / (self.df.loc[i, "SA"] * self.RHO_I)
                dfds.loc[i, "cdt"] *= 1 / (self.df.loc[i, "SA"] * self.RHO_I)

    dfds = dfds.set_index("When").resample("D").sum().reset_index()
    dfds["When"] = dfds["When"].dt.strftime("%b %d")

    dfds = dfds.rename(
        columns={
            "solid": "Ice",
            "ppt": "Snow",
            "melted": "Melt",
            "gas": "Vapour sub./evap.",
        }
    )
    dfds["Vapour cond./dep."] = dfds["dpt"] + dfds["cdt"]

    y2 = dfds[
        [
            "Ice",
            "Snow",
            "Vapour cond./dep.",
            "Vapour sub./evap.",
            "Melt",
        ]
    ]

    dfd = self.df.set_index("When").resample("D").mean().reset_index()
    dfd["When"] = dfd["When"].dt.strftime("%b %d")


    dfds2 = self.df.set_index("When").resample("D").mean().reset_index()
    dfds2["When"] = dfds2["When"].dt.strftime("%b %d")
    dfds2 = dfds2.set_index("When")
    dfds = dfds.set_index("When")
    y3 = dfds2["SA"]
    y4 = dfds2["iceV"]
    y0 = dfds["Discharge"] * self.TIME_STEP / (60*1000)

    z = dfd[["$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$", "$q_{G}$"]]

    ax0 = fig.add_subplot(5, 1, 1)
    ax0 = y0.plot.bar(
        y="Discharge", linewidth=0.5, edgecolor="black", color="#0C70DE", ax=ax0
    )
    ax0.xaxis.set_label_text("")
    ax0.set_ylabel("Discharge ($m^3$)")
    ax0.grid(axis="y", color="#0C70DE", alpha=0.3, linewidth=0.5, which="major")
    at = AnchoredText("(a)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    x_axis = ax0.axes.get_xaxis()
    x_axis.set_visible(False)
    ax0.add_artist(at)

    ax1 = fig.add_subplot(5, 1, 2)
    ax1 = z.plot.bar(stacked=True, edgecolor="black", linewidth=0.5, ax=ax1)
    ax1.xaxis.set_label_text("")
    ax1.grid(color="black", alpha=0.3, linewidth=0.5, which="major")
    plt.ylabel("Energy Flux [$W\\,m^{-2}$]")
    plt.legend(loc="upper center", ncol=6)
    # plt.ylim(-125, 125)
    x_axis = ax1.axes.get_xaxis()
    x_axis.set_visible(False)
    at = AnchoredText("(b)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax1.add_artist(at)

    ax2 = fig.add_subplot(5, 1, 3)
    y2.plot(
        kind="bar",
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        color=["#D9E9FA", CB91_Blue, "#EA9010", "#006C67", "#0C70DE"],
        ax=ax2,
    )
    plt.ylabel("Thickness ($m$ w. e.)")
    plt.xticks(rotation=45)
    plt.legend(loc="upper center", ncol=6)
    # ax2.set_ylim(-0.03, 0.03)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
    x_axis = ax2.axes.get_xaxis()
    x_axis.set_visible(False)
    at = AnchoredText("(c)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax2.add_artist(at)

    ax3 = fig.add_subplot(5, 1, 4)
    ax3 = y3.plot.bar(
        y="SA", linewidth=0.5, edgecolor="black", color="xkcd:grey", ax=ax3
    )
    ax3.xaxis.set_label_text("")
    ax3.set_ylabel("Surface Area ($m^2$)")
    ax3.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
    x_axis = ax3.axes.get_xaxis()
    x_axis.set_visible(False)
    at = AnchoredText("(d)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax3.add_artist(at)

    ax4 = fig.add_subplot(5, 1, 5)
    ax4 = y4.plot.bar(
        x="When", y="iceV", linewidth=0.5, edgecolor="black", color="#D9E9FA", ax=ax4
    )
    ax4.xaxis.set_label_text("")
    ax4.set_ylabel("Ice Volume($m^3$)")
    ax4.grid(axis="y", color="black", alpha=0.3, linewidth=0.5, which="major")
    at = AnchoredText("(e)", prop=dict(size=10), frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax4.add_artist(at)
    ax4.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax4.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        output + "paper_figures/Model_Output_" + self.trigger + ".jpg", dpi=300, bbox_inches="tight"
    )
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, sharex="col", sharey="row", figsize=(15, 12)
    )

    x = self.df.When

    y1 = self.df.a
    y2 = self.df.f_cone
    ax1.plot(x, y1, color="#16697a")
    ax1.set_ylabel("Albedo")
    ax1t = ax1.twinx()
    ax1t.plot(x, y2, color="#ff6d00", linewidth=0.5)
    ax1t.set_ylabel("$f_{cone}$", color="#ff6d00")
    for tl in ax1t.get_yticklabels():
        tl.set_color("#ff6d00")
    ax1.set_ylim([0, 1])
    ax1t.set_ylim([0, 1])

    y1 = self.df.T_s
    y2 = self.df.T_bulk
    ax2.plot(
        x, y1, "k-", linestyle="-", color="#00b4d8", linewidth=0.5, label="Surface"
    )
    ax2.set_ylabel("Temperature [$\\degree C$]")
    ax2.plot(x, y2, linestyle="-", color="#023e8a", linewidth=1, label="Bulk")
    ax2.set_ylim([-20, 1])
    ax2.legend()

    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    plt.savefig(
        output+ "paper_figures/albedo_temperature.jpg", dpi=300, bbox_inches="tight"
    )
    plt.close("all")

    self.df = self.df.rename(
        {
            "$q_{SW}$": "SW",
            "$q_{LW}$": "LW",
            "$q_S$": "Qs",
            "$q_L$": "Ql",
            "$q_{F}$": "Qf",
            "$q_{G}$": "Qg",
        },
        axis=1,
    )

