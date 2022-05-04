"""Function that returns returns new discharge after height increase by dh
"""
import sys, os
import pandas as pd
import math
import numpy as np
import logging
import coloredlogs
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
# from sklearn.linear_model import LinearRegression

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from src.utils.settings import config
from src.models.icestupaClass import Icestupa


def get_dis_new(
    dia=0.016, dh=1, dis=0
): 
    Area = math.pi * math.pow(dia, 2) / 4
    G = 9.8
    v= dis/ (60 * 1000 * Area)
    if v**2 - 2 * G * dh < 0 :
        dis_new = 0
    else:
        v_new = math.sqrt( v**2 - 2 * G * dh )
        dis_new = v_new * (60 * 1000 * Area)
    # logger.warning("Discharge calculated is %s" % (dis))
    return dis_new

def get_projectile(
    dia=0, h_f=3, dis=0, r=0, theta_f = 45
):  # returns discharge or spray radius using projectile motion

    Area = math.pi * math.pow(dia, 2) / 4
    theta_f = math.radians(theta_f)
    G = 9.8
    if r == 0:
        data_ry = []
        v = dis / (60 * 1000 * Area)
        t = 0.0
        while True:
            # now calculate the height y
            y = h_f + (t * v * math.sin(theta_f)) - (G * t * t) / 2
            # projectile has hit ground level
            if y < 0:
                break
            r = v * math.cos(theta_f) * t
            data_ry.append((r, y))
            t += 0.01
        # logger.warning("Spray radius is %s" % (r))
        return r
    else:
        v = math.sqrt(
            G ** 2
            * r ** 2
            / (math.cos(theta_f) ** 2 * 2 * G * h_f + math.sin(2 * theta_f) * G * r)
        )
        dis = v * (60 * 1000 * Area)
        # logger.warning("Discharge calculated is %s" % (dis))
        return dis


def get_dis_with_height(
    h_f, Q_i=11.3, h_i=4, dia=4.9 
):  # returns discharge
    Q_i /=(1000*60)
    dia /=(1000)
    if Q_i**2 + 2*9.8*(h_i-h_f) * (np.pi * dia ** 2/4)**2 > 0:
        Q_f = np.sqrt(Q_i**2 + 2*9.8*(h_i-h_f) * (np.pi * dia ** 2/4)**2)
    else:
        Q_f = 0

    Q_f*=1000*60
    Q_f=round(Q_f,2)

    return Q_f


if __name__ == "__main__":
    # Main logger
    logger = logging.getLogger(__name__)
    logger.setLevel("ERROR")

    # print(get_dis_new(dh=4, dis_old=60))
    # print(get_dis_new(dh=1, dis=13))
    # print(get_projectile(h_f=3, dia=0.006, r=3, theta_f=60))
    # print(get_projectile(h_f=4.7, dia=0.005, dis=5))
    # print(get_projectile(h_f=4, dia=0.0049, dis=5.63))
    # print(get_projectile(h_f=4, dia=0.0049, dis=11))
    # print(get_projectile(h_f=4.7, dia=0.0053, r=4.1))
    # print(get_projectile(h_f=5.7, dia=0.0049, dis=11))
    # print(get_dis_with_height(h_f=5.7))
    dis = []
    for dia_f in np.arange(5,8,0.5):
        for row in np.arange(1,10,0.05):
            Q_f = get_dis_with_height(h_f=row, dia = dia_f)
            dis.append([row, dia_f, Q_f])

    df = pd.DataFrame(dis, columns=["Height", "Diameter", "Discharge"])
    df.to_csv('/home/suryab/work/air_model/data/guttannen22/interim/dis_height.csv', index=False)

    x = np.array([5, 19]).reshape((-1, 1))
    y = np.array([4300,0])

    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    y_pred = model.predict([[6]])
    print('predicted response:', y_pred, sep='\n')
    pres = []
    for dia in np.arange(1,20):
        y_pred = model.intercept_ + model.coef_ * dia 
        y_pred = y_pred[0]/1000
        pres.append([dia, y_pred])
    df2 = pd.DataFrame(pres, columns=["Diameter", "Pressure"])
    # df2 = pd.DataFrame(pres)
    df2.to_csv('/home/suryab/work/air_model/data/guttannen22/interim/dia_pressure.csv', index=False)



