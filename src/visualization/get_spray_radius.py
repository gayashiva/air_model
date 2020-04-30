import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from src.models.air import Icestupa
import math

# fig, ax = plt.subplots()

schwarzsee = Icestupa()

v_old = schwarzsee.read_input()
aperture_f_old = 5e-03
h_old = 1.35
print(v_old)

aperture_f_new = np.arange(3.6, 5, 0.1).tolist()

h_new = []
r_new = []

for i in aperture_f_new:
    i = i/1000
    v_new = (math.pi * aperture_f_old**2 * v_old / (i **2 * math.pi))

    h_new.append(h_old-(v_new ** 2 - v_old**2) / (2*9.81))
    r_new.append(schwarzsee.projectile_xy(v=v_new, h=h_new[-1]))

print(h_new)
# r_new = schwarzsee.projectile_xy(v=v_new, h=h_new)

fig, ax = plt.subplots()

ax.scatter(aperture_f_new ,h_new)

plt.show()

fig, ax = plt.subplots()

ax.scatter(aperture_f_new ,r_new)

plt.show()