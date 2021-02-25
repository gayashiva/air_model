import numpy as np; import matplotlib.pyplot as plt
import pandas as pd

dir = "/home/surya/Programs/PycharmProjects/air_model/data/raw/"

def lum2temp(y0):

	df_in = pd.read_csv(dir + "lum_values.csv", sep=",")

	# Correct values
	mask = (df_in["X"]<2000)
	df_in= df_in[mask]

	k = df_in.loc[df_in["Y"] == df_in["Y"].max(), "X"].values

	# Correct values
	mask = (df_in["X"]<k[0])
	df_in= df_in[mask]

	x = df_in.X
	y = df_in.Y

	h = df_in.loc[df_in["Y"] == 200, "X"].values

	x1 = x[:h[0]]
	y1 = y[:h[0]]
	A1 = np.vstack([x1, np.ones(len(x1))]).T
	m1, c1 = np.linalg.lstsq(A1, y1, rcond=None)[0]

	x2 = x[h[0]:]
	y2 = y[h[0]:]
	A2 = np.vstack([x2, np.ones(len(x2))]).T
	m2, c2 = np.linalg.lstsq(A2, y2, rcond=None)[0]

	if y0 >= 200:
		x0 = (y0-c2)/m2
	else:
		x0 = (y0-c1)/m1

	return x0