import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import math
import time
from tqdm import tqdm
import shutil, os, sys
import glob
import fnmatch
from os import listdir
from os.path import isfile, join
import shutil

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(dirname)

site = 'guttannen22'
oldpath = "/home/suryab/switchdrive/Icestupas/Guttannen/timelapse/"+ site + "/"
newpath = "/home/suryab/Pictures/timelapses/" + site + "/"
# oldpath = "/home/suryab/Pictures/rain_event/raw/"
# newpath = "/home/suryab/Pictures/rain_event/interim/"
onlyfiles = [f for f in listdir(oldpath) if isfile(join(oldpath, f))]
df_names = pd.DataFrame({"col": onlyfiles})

df_names["Label"] = df_names["col"].str.split("m").str[-1]

df_names["Label"] = (
    "20"
    + df_names["Label"].str[0:2]
    + "-"
    + df_names["Label"].str[2:4]
    + "-"
    + df_names["Label"].str[4:6]
    + " "
    + df_names["Label"].str[6:8]
)
# print(df_names.Label)

df_names["When"] = pd.to_datetime(df_names["Label"], format="%Y-%m-%d %H")
df_names = df_names.set_index("When").sort_index().reset_index()
print(df_names.head())

if not os.path.exists(newpath):
    os.mkdir(newpath)
else:
    # Delete folder contents
    folder = newpath
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

# df_names["When"] = df_names["When"].dt.strftime("%b %d %H")

# df_names["When"] = pd.to_datetime(df_names["When"], format="%b %d %H")

print(df_names.head())

for i in range(0, df_names.shape[0]):

    if 6 < df_names.loc[i, "When"].hour < 19:
        if df_names.loc[i, "When"].hour % 4 == 0:
            shutil.copy(oldpath + df_names.loc[i, "col"], newpath)
            os.rename(
                newpath + df_names.loc[i, "col"],
                newpath
                + str(i)
                + "_"
                + str(df_names.loc[i, "When"].strftime("%b-%d %H") + ":00"),
            )
