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

discharge = 3.34

# aperture_f_old = 5e-03
# h_old = 1.35
# v_old = discharge / (60 * 1000 * aperture_f_old**2/4 * math.pi)


# aperture_f_old = 10e-03
# h_old = 1.35
# v_old = discharge / (60 * 1000 * aperture_f_old**2/4 * math.pi)


aperture_f_old = 5e-03
v_old = discharge / (60 * 1000 * aperture_f_old**2/4 * math.pi)
h_old = 1.35


results = []

h_new = 0.1

aperture_f_new = aperture_f_old
while aperture_f_new > 2e-03:

    v_new = (math.pi * aperture_f_old ** 2 * v_old / (aperture_f_new ** 2 * math.pi))
    h_new = h_old
    r_new = schwarzsee.projectile_xy(v=v_new, h=h_old)
    results.append(pd.Series([aperture_f_new * 1000,
                              v_new,
                              h_new,
                              r_new]
                             ))
    aperture_f_new = aperture_f_new - 1e-04

# """For discharge constant"""
# for i in aperture_f_new:
#     i = i/1000
#     v_new = (math.pi * aperture_f_old**2 * v_old / (i**2 * math.pi))
#
#     h_new = h_old-(v_new ** 2 - v_old**2) / (2*9.81)
#     r_new = schwarzsee.projectile_xy(v=v_new, h=h_new)
#
#     results.append(pd.Series([i * 1000,
#                         v_new,
#                         h_new,
#                         r_new]
#                        ))

results = pd.DataFrame(results)
results = results.rename(
    columns={0: 'aperture_dia', 1: 'droplet_speed', 2: 'height', 3: 'spray_radius'})

print(results)

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.plot(results["aperture_dia"], results["spray_radius"], "k-")
ax1.set_xlabel("Aperture diameter [$mm$]")
ax1.set_ylabel("Spray Radius [$m$]")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.savefig("/home/surya/Programs/PycharmProjects/air_model/data/processed/schwarzsee/simulations/change_fountain.jpg", dpi=150, bbox_inches="tight")
plt.show()