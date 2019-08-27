#!/usr/bin/env python

from manimlib.imports import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource
import math
import sys
import os


# To watch one of these scenes, run the following:
# python -m manim C:\air_model\src\visualization\visualize.py Forecast -pl
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

        site = input("Input the Field Site Name: ") or 'schwarzsee'

        dirname = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

        input_folder = os.path.join(dirname, "data/processed/" )

        # output_folder = os.path.join(dirname, "data/processed/" )

        # read files
        filename0 = os.path.join(input_folder, site + "_model_gif.csv")
        df= pd.read_csv(filename0)
        df['When'] = pd.to_datetime(df['When'], format = '%Y.%m.%d %H:%M:%S')
        df=df.round(2)

        df2 = df
        df2= df2.set_index('When').resample('D').mean()
        df2= df2.resample('5T').ffill().reset_index()
        df2=df2.round(2)
        print(df2.head())

        text0 = TexMobject(site + " -"  +  " IceStupa",aligned_edge=TOP)
        text0.to_edge(UP)

        text3 = TexMobject("Temperature = ",aligned_edge=TOP)
        text3.scale(0.5)
        text3.to_edge(DOWN+LEFT)

        text4 = TexMobject("Ice" + "-"  + "left ",aligned_edge=TOP)
        text4.to_edge(DOWN)

        self.add(text0, text3)

        h_f = 0

        if site == 'schwarzsee' :
            scale_factor = 1.35
        else:
            scale_factor = 3

        for i in range(1,df2.shape[0], 144):

            date = TextMobject(str(df.loc[i, 'When'].date()))
            date.scale(h_f/scale_factor)
            date.next_to(text0, DOWN)

            dimensions= TexMobject(str(df2.loc[i, 'r_ice']*2)+ "  m"+ " \\cross" + str(df2.loc[i, 'h_ice']*2)+ "  m" ,aligned_edge=TOP)
            dimensions.scale(h_f/scale_factor)
            dimensions.next_to(text4 ,UP)


            triangle = Polygon(np.array([0, 1 * df2.loc[ i, 'h_ice'] ,0]),np.array([ 1 * df2.loc[i, 'r_ice'] ,0,0]),np.array([-1 * df2.loc[i, 'r_ice'] ,0,0]))
            triangle.set_fill(WHITE, opacity=1)
            triangle.scale(h_f/scale_factor)
            triangle.next_to(dimensions ,UP)

            temp = TexMobject( str(df2.loc[i, 'T_a'])+ "  C",aligned_edge=TOP)
            temp.scale(h_f/scale_factor)
            temp.next_to(text3,RIGHT)

            ice = TexMobject( str(df2.loc[i, 'ice']) + " l",aligned_edge=TOP, color=WHITE)
            ice.next_to(text4,RIGHT)

            if df.loc[i, 'h_f'] != h_f:

                print(df2.loc[i, 'When'])
                h_f=df.loc[i, 'h_f']

                if df.loc[i,'Discharge'] > 0:
                    height = Vector(direction = UP, color=BLUE, buff = 0)
                else:
                    print("black")
                    height = Vector(direction = UP, color=BLACK, buff = 0)


                height.scale(h_f/scale_factor * 3/5 * h_f)

                height.next_to(triangle, UP)

            self.add(triangle, height, date, dimensions, temp, text4, ice)
            self.wait(0.25)
            self.remove( triangle, height, date, dimensions, temp, text4, ice)
#
# class Schwarzsee(Scene):
#     def construct(self):
#
#         # read files
#         df= pd.read_csv('C:/Users/Balasubr/Dropbox/Surya/Modelling_AIR/Data_Analysis/Sites/Schwarzsee/output_data/model_gif_10.csv')
#         df['When'] = pd.to_datetime(df['When'], format = '%Y.%m.%d %H:%M:%S')
#         df=df.round(2)
#
#         df2 = df
#         df2= df2.set_index('When').resample('D').mean()
#         df2= df2.resample('5T').ffill().reset_index()
#         df2=df2.round(2)
#         print(df2.head())
#
#         text0 = TexMobject("Schwarzsee"  + "-"  +  " IceStupa",aligned_edge=TOP)
#         text0.to_edge(UP)
#
#         text3 = TexMobject("Temperature = ",aligned_edge=TOP)
#         text3.scale(0.5)
#         text3.to_edge(DOWN+LEFT)
#
#
#         text4 = TexMobject("Ice" + "-"  + "left ",aligned_edge=TOP)
#         text4.to_edge(DOWN)
#
#
#         self.add(text0, text3)
#
#         for i in range(1,df2.shape[0], 144):
#
#             date = TextMobject(str(df.loc[i, 'When'].date()))
#             date.next_to(text0, DOWN)
#
#
#             dimensions= TexMobject(str(df2.loc[i, 'r_ice']*2)+ "  m"+ " \\cross" + str(df2.loc[i, 'h_ice']*2)+ "  m" ,aligned_edge=TOP)
#             dimensions.next_to(text4 ,UP)
#             dimensions.scale(0.5)
#
#             triangle = Polygon(np.array([0, 1 * df2.loc[ i, 'h_ice'] ,0]),np.array([ 1 * df2.loc[i, 'r_ice'] ,0,0]),np.array([-1 * df2.loc[i, 'r_ice'] ,0,0]))
#             triangle.set_fill(WHITE, opacity=1)
#             triangle.scale(2)
#             triangle.next_to(dimensions ,UP)
#
#
#             temp = TexMobject( str(df2.loc[i, 'T_a'])+ "  C",aligned_edge=TOP)
#             temp.scale(0.5)
#             temp.next_to(text3,RIGHT)
#
#
#             ice = TexMobject( str(df2.loc[i, 'ice']) + " l",aligned_edge=TOP, color=WHITE)
#             ice.next_to(text4,RIGHT)
#
#             if not df2.loc[i,'Discharge'] > 0 :
#                 dot = Dot(point= np.array([0,1,0]), color=WHITE)
#             else:
#                 dot = Dot(point= np.array([0,1,0]), color=BLUE)
#
#             dot.to_edge(UP+LEFT)
#
#             self.add(triangle, date, dimensions, temp, text4, ice)
#             self.wait(0.5)
#             self.remove( triangle, date, dimensions, temp, text4, ice)
