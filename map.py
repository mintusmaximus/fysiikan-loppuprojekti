from math import asin, cos, radians, sin, sqrt

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

dfLoc = pd.read_csv("./projectData/Location.csv")
dfAccel = pd.read_csv("./projectData/Linear Acceleration.csv")


# "Time (s)","Linear Acceleration x (m/s^2)","Linear Acceleration y (m/s^2)","Linear Acceleration z (m/s^2)"
# "Time (s)","Latitude (°)","Longitude (°)","Height (m)","Velocity (m/s)","Direction (°)","Horizontal Accuracy (m)","Vertical Accuracy (m)"


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


dfLoc["Distance_calc"] = np.zeros(len(dfLoc))
for i in range(len(dfLoc) - 1):
    lon1 = dfLoc["Longitude (°)"][i]
    lon2 = dfLoc["Longitude (°)"][i + 1]
    lat1 = dfLoc["Latitude (°)"][i]
    lat2 = dfLoc["Latitude (°)"][i + 1]
    dfLoc.loc[i + 1, "Distance_calc"] = haversine(lon1, lat1, lon2, lat2)
dfLoc["Total_distance"] = dfLoc["Distance_calc"].cumsum()


st.title("Fysiikan lopputyö")
st.text("Kokonaismatka metreinä: " + str(dfLoc["Total_distance"].max() * 1000)[:5])
st.text("Keskinopeus m/s: " + str(dfLoc["Velocity (m/s)"].mean())[:3])
st.text("Keskinopeus km/h: " + str(dfLoc["Velocity (m/s)"].mean() * 3.6)[:3])

start_lat = dfLoc["Latitude (°)"].mean()
start_long = dfLoc["Longitude (°)"].mean()
my_map = folium.Map(location=[start_lat, start_long], zoom_start=14)

folium.PolyLine(
    dfLoc[["Latitude (°)", "Longitude (°)"]], color="blue", weight=5, opacity=1
).add_to(my_map)
st_map = st_folium(my_map, width=900, height=500)
