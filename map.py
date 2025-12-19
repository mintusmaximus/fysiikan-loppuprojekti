from math import asin, cos, radians, sin, sqrt

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

df = pd.read_csv("./projectData/Location.csv")

st.title("Accuracy over time")
st.line_chart(
    df[df["Horizontal Accuracy (m)"] < 50],
    x="Time (s)",
    y="Horizontal Accuracy (m)",
    x_label="Time (s)",
    y_label="Horizontal Accuracy (m)",
)


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


# lasketaan matka
df["Distance_calc"] = np.zeros(len(df))
for i in range(len(df) - 1):
    lon1 = df["Longitude (°)"][i]
    lon2 = df["Longitude (°)"][i + 1]
    lat1 = df["Latitude (°)"][i]
    lat2 = df["Latitude (°)"][i + 1]
    df.loc[i + 1, "Distance_calc"] = haversine(lon1, lat1, lon2, lat2)

df["Total_distance"] = df["Distance_calc"].cumsum()

fig, ax = plt.subplots(figsize=(12, 5))
plt.plot(df["Time (s)"], df["Total_distance"])
plt.ylabel("testing")
plt.xlabel("testing rebuild")
st.pyplot(fig)


start_lat = df["Latitude (°)"].mean()  # lat keskiarvo
start_long = df["Longitude (°)"].mean()  # lon keskiarvo
my_map = folium.Map(
    location=[start_lat, start_long], zoom_start=14
)  # luodaan kartta keskiarvoista

folium.PolyLine(
    df[["Latitude (°)", "Longitude (°)"]], color="blue", weight=5, opacity=1
).add_to(my_map)
st_map = st_folium(my_map, width=900, height=650)
