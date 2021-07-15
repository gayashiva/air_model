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
        self.df = self.df[1:i]
        self.df = self.df.reset_index(drop=True)
    break
