from math import asin, cos, radians, sin, sqrt

import folium
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, filtfilt
from streamlit_folium import st_folium

dfLoc = pd.read_csv(
    "https://raw.githubusercontent.com/mintusmaximus/fysiikan-loppuprojekti/refs/heads/master/projectData/Location.csv"
)
dfAccel = pd.read_csv(
    "https://raw.githubusercontent.com/mintusmaximus/fysiikan-loppuprojekti/refs/heads/master/projectData/Linear%20Acceleration.csv"
)


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


sig = dfAccel["Linear Acceleration y (m/s^2)"]
t = dfAccel["Time (s)"]

N = len(sig)
dt = np.max(t) / N

fourier = np.fft.fft(sig, N)
psd = fourier * np.conj(fourier) / N
freq = np.fft.fftfreq(N, dt)

positive = freq >= 0
freq_pos = freq[positive]
psd_pos = psd[positive].real

freq = freq_pos <= 6
freq_crop = freq_pos[freq]
psd_crop = psd_pos[freq]

fourier_clean = fourier.copy()
fourier_clean[psd.real < 500] = 0
sig_clean = np.fft.ifft(fourier_clean)

step = 0

for i in range(N - 1):
    if sig_clean[i] / sig_clean[i + 1] < 0:
        step += 1

step = step / 2


data = dfAccel["Linear Acceleration y (m/s^2)"]


def butter_lp(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    y = filtfilt(b, a, data)
    return y


timediff = dfAccel["Time (s)"].diff().dropna()
cutoff = 5.0
fs = 1.0 / timediff.iloc[0]
nyq = fs / 2
sig = data
data_filt = butter_lp(sig, cutoff, nyq, order=3)
filtstep = 0
for i in range(len(data) - 1):
    if data_filt[i] / data_filt[i + 1] < 0:
        filtstep += 1

filtstep = filtstep / 2

maxdist = dfLoc["Total_distance"].max() * 1000


st.title("Fysiikan lopputyö")
st.text(
    "Kokonaismatka metreinä: " + str(dfLoc["Total_distance"].max() * 1000)[:5] + "m"
)
st.text("Keskinopeus m/s: " + str(dfLoc["Velocity (m/s)"].mean())[:3])
st.text("Keskinopeus km/h: " + str(dfLoc["Velocity (m/s)"].mean() * 3.6)[:3])
st.text("Askelmäärä laskettu Fourier-analyysillä on: " + str(step))
st.text("Askelmäärä laskettu suodatuksella on: " + str(filtstep))
st.text(
    "Askelpituus laskettu fourier-suodatuksesta on: " + str((maxdist) / step)[:3] + " m"
)
st.text(
    "Askelpituus laskettu suodatuksesta on: " + str((maxdist) / filtstep)[:3] + " m"
)

st.text(
    "Signaalit on rajattu ensimmäiseen 10 sekuntiin sovelluksen suorituskykyä varten"
)
dfAccel_cropped = dfAccel[dfAccel["Time (s)"] <= 10]
st.subheader("Käytettävä mittaus - Linear Acceleration y (m/s^2), rajattu")
st.line_chart(
    data=dfAccel_cropped,
    x="Time (s)",
    y=["Linear Acceleration y (m/s^2)"],
)

st.subheader("Tehospektri")
df_psd = pd.DataFrame({"Taajuus": freq_crop, "Teho": psd_crop})
st.line_chart(data=df_psd, x="Taajuus", y="Teho")

st.subheader("Fourier-analyysillä suodatettu signaali, rajattu")
df_filt_cropped = pd.DataFrame(
    {"Time (s)": dfAccel["Time (s)"], "Filtered signal": sig_clean.real}
)
st.line_chart(
    data=df_filt_cropped[df_filt_cropped["Time (s)"] <= 10],
    x="Time (s)",
    y="Filtered signal",
)

st.subheader("Scipy-suodattimella suodatettu signaali, rajattu")
df_filt_cropped2 = pd.DataFrame(
    {"Time (s)": dfAccel["Time (s)"], "Filtered signal": data_filt}
)
st.line_chart(
    data=df_filt_cropped2[df_filt_cropped2["Time (s)"] <= 10],
    x="Time (s)",
    y="Filtered signal",
)

st.subheader("Kartta")
start_lat = dfLoc["Latitude (°)"].mean()
start_long = dfLoc["Longitude (°)"].mean()
my_map = folium.Map(location=[start_lat, start_long], zoom_start=14)

folium.PolyLine(
    dfLoc[["Latitude (°)", "Longitude (°)"]], color="blue", weight=5, opacity=1
).add_to(my_map)
st_map = st_folium(my_map, width=900, height=500)
