import pandas as pd
import math
import numpy as np
from functools import lru_cache

import logging
import coloredlogs

# Required for colored logging statements
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
coloredlogs.install(
    # fmt="%(name)s %(levelname)s %(message)s",
    fmt="%(levelname)s %(message)s",
    logger=logger,
)


@lru_cache
def get_discharge(self):  # Provides discharge info based on trigger setting

    self.df["Discharge"] = 0

    if self.trigger == "Temperature":
        self.df["Prec"] = 0
        mask = (self.df["T_a"] < self.crit_temp) & (self.df["SW_direct"] < 100)
        mask_index = self.df[mask].index
        self.df.loc[mask_index, "Discharge"] = 1 * self.discharge

        logger.debug(
            f"Hours of spray : %.2f"
            % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
        )

    if self.trigger == "NetEnergy":

        col = [
            "T_s",  # Surface Temperature
            "T_bulk",  # Bulk Temperature
            "f_cone",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "ppt",
            "dpt",
            "cdt",
        ]

        for column in col:
            self.df[column] = 0

        self.df.Discharge = 0
        logger.debug("Calculating discharge from energy trigger ...")
        for row in tqdm(self.df[1:-1].itertuples(), total=self.df.shape[0]):
            self.energy_balance(row, mode="trigger")
        mask = self.df["TotalE"] < 0
        mask_index = self.df[mask].index
        self.df.loc[mask_index, "Discharge"] = 1 * self.discharge

        col = [
            "T_s",  # Surface Temperature
            "T_bulk",  # Bulk Temperature
            "f_cone",
            "TotalE",
            "SW",
            "LW",
            "Qs",
            "Ql",
            "Qf",
            "Qg",
            "ppt",
            "dpt",
            "cdt",
        ]
        self.df.drop(columns=col)

        logger.debug(
            f"Hours of spray : %.2f"
            % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
        )

    if self.trigger == "Manual" and self.name == "guttannen":
        self.df["Discharge"] = self.discharge

    if self.trigger == "Manual" and self.name == "schwarzsee":

        mask = self.df["When"] >= self.start_date
        self.df = self.df.loc[mask]
        self.df = self.df.reset_index(drop=True)
        logger.warning(f"Start date changed to %s" % (self.start_date))

        df_f = pd.read_csv(
            os.path.join(dirname, "data/" + "schwarzsee" + "/interim/")
            + "schwarzsee_input_field.csv"
        )
        df_f["When"] = pd.to_datetime(df_f["When"], format="%Y.%m.%d %H:%M:%S")
        df_f = (
            df_f.set_index("When").resample(str(int(self.TIME_STEP / 60)) + "T").mean()
        )
        self.df = self.df.set_index("When")
        mask = df_f["Discharge"] != 0
        f_on = df_f[mask].index
        self.df.loc[f_on, "Discharge"] = df_f["Discharge"]
        self.df = self.df.reset_index()
        self.df["Discharge"] = self.df.Discharge.replace(np.nan, 0)
        logger.debug(
            f"Hours of spray : %.2f"
            % (self.df.Discharge.astype(bool).sum(axis=0) * self.TIME_STEP / 3600)
        )

    mask = self.df["When"] > self.fountain_off_date
    mask_index = self.df[mask].index
    self.df.loc[mask_index, "Discharge"] = 0
