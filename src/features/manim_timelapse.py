from manim import *
import pandas as pd
import numpy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
import math
import sys
import os

# To watch one of these scenes, run the following:
# python -m manim C:\air_model\src\features\timelapse.py Forecast -pl
#
# Use the flat -l for a faster rendering at a lower
# quality.
# Use -s to skip to the end and just save the final frame
# Use the -p to have the animation (or image, if -s was
# used) pop up once done.
# Use -n <number> to skip ahead to the n'th animation of a scene.
# Use -r <number> to specify a resolution (for example, -r 1080
# for a 1920x1080 video)


class Forecast(Scene):
    def construct(self):

        site = config.output_file
        trigger = "Manual"

        dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        input_folder = os.path.join(dirname, "data/" + site + "/processed")

        # read files
        filename0 = os.path.join(input_folder, site + "_manim_"+ trigger + ".csv")
        df = pd.read_csv(filename0)
        df["When"] = pd.to_datetime(df["When"], format="%Y.%m.%d %H:%M:%S")
        df = df.round(2)

        df2 = df
        df2 = df2.set_index("When").resample("D").mean()
        df2 = df2.round(2).reset_index()
        print(df2.head())

        text0 = MathTex(site + " -" + " IceStupa")
        text0.to_edge(UP)

        text3 = MathTex("Temperature = ")
        text3.scale(0.5)
        text3.to_edge(DOWN + LEFT)

        text4 = MathTex("Ice" + "-" + "left ")
        text4.to_edge(DOWN)

        self.add(text0, text3)

        h_f = 0

        if site == "schwarzsee19":
            scale_factor = 1.35
            scale_factor_shape = 1.35 / 4
        else:
            scale_factor = df2.r_ice.max() *3
            scale_factor_shape = df2.r_ice.max()

        # hours = 4

        # for i in range(0, df2.shape[0] - 150, 12 * hours):
        for i in range(0, df2.shape[0]):

            date = Tex(str(df.loc[i, "When"].date()))
            # date.scale(h_f / scale_factor)
            date.next_to(text0, DOWN)

            dimensions = MathTex(
                str(df2.loc[i, "r_ice"] * 2)
                + "  m"
                + " \\cross"
                + str(df2.loc[i, "h_ice"] * 2)
                + "  m",
                # aligned_edge=TOP,
            )
            # dimensions.scale(h_f / scale_factor)
            dimensions.next_to(text4, UP)

            triangle = Polygon(
                np.array([0, 1 * df2.loc[i, "h_ice"], 0]),
                np.array([1 * df2.loc[i, "r_ice"], 0, 0]),
                np.array([-1 * df2.loc[i, "r_ice"], 0, 0]),
            )
            triangle.set_fill(WHITE, opacity=1)
            triangle.scale(h_f / scale_factor_shape)
            triangle.next_to(dimensions, UP)

            # temp = MathTex(str(df2.loc[i, "T_a"]) + "  C", aligned_edge=TOP)
            temp = MathTex(str(df2.loc[i, "T_a"]) + "  C")
            # temp.scale(h_f / scale_factor)
            temp.next_to(text3, RIGHT)

            ice = MathTex(
                # str(df2.loc[i, "ice"]) + " l", aligned_edge=TOP, color=WHITE
                str(df2.loc[i, "ice"]) + " l",
                color=WHITE,
            )
            ice.next_to(text4, RIGHT)

            if df.loc[i, "h_f"] != h_f:

                # print(df2.loc[i, "When"], df.loc[i, "h_f"])
                h_f = df.loc[i, "h_f"]

            if df.loc[i, "Discharge"] > 0:
                height = Vector(direction=UP, color=BLUE, buff=0)
            else:
                height = Vector(direction=UP, color=BLACK, buff=0)

            height.scale(h_f / scale_factor * 3 / 5 * h_f)

            height.next_to(triangle, UP)

            self.add(triangle, height, date, dimensions, temp, text4, ice)
            self.wait(0.25)
            self.remove(triangle, height, date, dimensions, temp, text4, ice)
