
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)

from src.utils.settings import config
from src.models.methods.metadata import get_parameter_metadata
from src.models.icestupaClass import Icestupa
from src.models.methods.metadata import get_parameter_metadata

location = 'guttannen21'
SITE, FOLDER = config(location)
icestupa = Icestupa(location)
icestupa.read_output()

icestupa.df = icestupa.df.rename(
    {
        "SW": "$q_{SW}$",
        "LW": "$q_{LW}$",
        "Qs": "$q_S$",
        "Ql": "$q_L$",
        "Qf": "$q_{F}$",
        "Qg": "$q_{G}$",
        "Qsurf": "$q_{surf}$",
        "Qmelt": "$-q_{freeze/melt}$",
        "Qt": "$-q_{T}$",
    },
    axis=1,
)

dfd = icestupa.df.set_index("When").resample("D").mean().reset_index()
dfd[["$-q_{freeze/melt}$", "$-q_{T}$"]] *=-1
z = dfd[["$-q_{freeze/melt}$",  "$q_{SW}$", "$q_{LW}$", "$q_S$", "$q_L$", "$q_{F}$","$-q_{T}$", "$q_{G}$"]]

z.index = z.index + 1
days = 20

fig,(ax,ax2) = plt.subplots(1, 2, sharey=True)

# plot the same data on both axes
# ax.plot(x, y, 'bo')
# idx_slice = z.index < days
z.plot.bar(
    stacked=True, 
    edgecolor="black", 
    linewidth=0.2, 
    # color=[purple, red, orange, green, "xkcd:yellowgreen", "xkcd:azure", pink,blue ],
    ax=ax
    )
z.plot.bar(
    stacked=True, 
    edgecolor="black", 
    linewidth=0.2, 
    # color=[purple, red, orange, green, "xkcd:yellowgreen", "xkcd:azure", pink,blue ],
    ax=ax2 
    )

# zoom-in / limit the view to different portions of the data
ax.set_xlim(0,days) # most of the data
ax2.set_xlim(z.shape[0]-days,z.shape[0]) # outliers only

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax2.yaxis.tick_right()

# Make the spacing between the two axes a bit smaller
plt.subplots_adjust(wspace=0.15)

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1). Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
ax.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-left diagonal

kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
ax2.plot((-d,d),(-d,+d), **kwargs) # top-right diagonal
ax2.plot((-d,d),(1-d,1+d), **kwargs) # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.savefig(
    "data/paper/mass_energy_bal.jpg",
    dpi=300,
    bbox_inches="tight",
)
