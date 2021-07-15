"""Icestupa class function that checks stop and saves results
"""

# External modules
import math, json
import numpy as np
import logging
import sys

# Module logger
logger = logging.getLogger(__name__)

def stop_model(self,i,all_cols, stop=0):
    logger.error(
        "Simulation ends %s %0.1f " % (self.df.When[i], self.df.iceV[i])
    )

    if (
        self.df.loc[i - 1, "When"] < self.fountain_off_date
        and self.df.loc[i - 1, "melted"] > 0
    ):
        logger.error("Skipping %s" % self.df.loc[i, "When"])
        stop = 1
        # break
    else:

        col_list = [
            "dep",
            "ppt",
            "fountain_froze",
            "fountain_runoff",
            "sub",
            "melted",
        ]
        for column in col_list:
            self.df.loc[i - 1, column] = 0

        last_hour = i-1
        # self.df = self.df[1:i]
        # self.df = self.df.reset_index(drop=True)

        iceV_max = round(self.df[:last_hour]["iceV"].max(), 1)
        M_input = round(self.df["input"].iloc[last_hour], 1)
        M_F = round(
            self.df[:last_hour]["Discharge"].sum() * self.DT / 60
            + self.df.loc[0, "input"]
            - self.V_dome * self.RHO_I,
            1,
        )
        M_ppt = round(self.df[:last_hour]["ppt"].sum(), 1)
        M_dep = round(self.df[:last_hour]["dep"].sum(), 1)
        M_water = round(self.df["meltwater"].iloc[last_hour], 1)
        M_runoff = round(self.df["unfrozen_water"].iloc[last_hour], 1)
        M_sub = round(self.df["vapour"].iloc[last_hour], 1)
        M_ice = round(self.df["ice"].iloc[last_hour] - self.V_dome * self.RHO_I, 1)

        results_dict = {}
        results = ["iceV_max", "last_hour", "M_input","M_F", "M_ppt", "M_dep", "M_water", "M_runoff", "M_sub", "M_ice"]

        for variable in results:
            results_dict[variable] = eval(variable)

        print("Summary of results for %s :"%self.name)
        print()
        for var in sorted(results_dict.keys()):
            print("\t%s: %r" % (var, results_dict[var]))
        print()

        with open(self.output + 'results.json', 'w') as fp:
            json.dump(results_dict, fp, sort_keys=True, indent=4)

        if last_hour > self.total_hours:
            self.df = self.df[:self.total_hours]
        else:
            for j in range(last_hour, self.total_hours):
                for col in all_cols:
                    self.df.loc[j, col] = 0
                    if col in ["iceV"]:
                        self.df.loc[j, col] = self.V_dome

        self.df = self.df.reset_index(drop=True)

        # Full Output
        filename4 = self.output + "model_output.csv"
        self.df.to_csv(filename4, sep=",")
        self.df.to_hdf(
            self.output + "model_output.h5",
            key="df",
            mode="a",
        )
        break
