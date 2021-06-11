"""Function that returns dataframe after changing its frequency
"""

import pandas as pd
import numpy as np
from pandas.tseries.frequencies import to_offset

import logging
import coloredlogs

logger = logging.getLogger(__name__)

def change_freq(self):  
    old_time_step = str(pd.to_timedelta(to_offset(pd.infer_freq(self.df["When"]))).seconds/60)
    new_time_step = str(self.DT/60)

    if new_time_step != old_time_step:
        self.df= self.df.set_index('When')
        dfx = self.df.missing_type.resample(old_time_step+'T').first()
        dfh = self.df.h_f.resample(old_time_step+'T').first()
        self.df= self.df.resample(old_time_step+'T').mean()
        self.df["missing_type"] = dfx
        self.df["h_f"] = dfh
        self.df= self.df.reset_index()
        logger.warning(f"Time steps changed from %s -> %s minutes" % (old_time_step,new_time_step ))
