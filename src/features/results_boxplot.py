# libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os,sys
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axisartist.axislines import Axes
from mpl_toolkits import axisartist
 
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # sns.set(style="darkgrid")
#     df = sns.load_dataset('tips')
#     print(df.head())
#     print(df.describe())
# 
#     sns.boxplot(x="day", y="total_bill", hue="smoker", data=df, palette="Set1", width=0.5)
#     plt.savefig("data/paper/box.jpg", bbox_inches="tight", dpi=300)
    locations = ['guttannen21',  'gangles21','guttannen20', 'schwarzsee19']

    df = pd.read_csv("data/paper/results.csv", parse_dates=['When'])
    print(df.head())
    print(df.describe())
    sns.set(style="darkgrid")
    sns.boxplot(x="day", y="total_bill", hue="smoker", data=df, palette="Set1", width=0.5)
    plt.savefig("data/paper/box.jpg", bbox_inches="tight", dpi=300)
