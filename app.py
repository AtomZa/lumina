import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import plotly.express as px
import plotly.subplots as subplots

import glob
import  os

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Setting
st.set_option('deprecation.showPyplotGlobalUse', False)

# Navigation
rad = st.sidebar.radio("Menu",["Preprocessing","Analysis","Result"])

# Add a file uploader to the sidebar
uploaded_folder = st.sidebar.text_input("Hi")

# Print the uploaded folder
if uploaded_folder is not None:
    st.write(f"Selected folder: {uploaded_folder}")

# Parameter 1
st.sidebar.subheader("Smooth & Interpolate")
ns = st.sidebar.number_input("Smooth Factor", min_value = 0, value = 100, step = 10)
ir = st.sidebar.number_input("Interpolate Interval", min_value = 0.000, value = 0.001, step = 0.001, format="%.3f")

# Parameter 2
st.sidebar.subheader("Time Interval")
c, d, e = st.columns(3)
starttime = st.sidebar.number_input("Start Time", min_value = 0, value = 1, step = 1)
timeinterval = st.sidebar.number_input("Time Interval", min_value = 0, value = 5, step = 1)
endtime = st.sidebar.number_input("End Time", min_value = 0, value = 26, step = 1)
timeaxis = np.arange(starttime, endtime+timeinterval, timeinterval)

# Parameter 3
st.sidebar.subheader("Wavelength")
minwavelength = st.sidebar.number_input("Min Wavelength", min_value = 0, value = 200, step = 10)
maxwavelength = st.sidebar.number_input("Max Wavelength", min_value = 0, value = 1000, step = 10)

if rad == "Preprocessing":

    # Check if a folder was uploaded
    if uploaded_folder is None:
        st.write("Please select a folder")
    else:
        # List all of the CSV files in the folder
        filelist = glob.glob(os.path.join(uploaded_folder, '*.csv'))

        # Create an empty list to store the plots
        plots = []

        for idx, uploaded_file in enumerate(filelist):
            if idx < starttime:
                pass
            elif idx > endtime:
                break
            elif (idx-1) % timeinterval != 0:
                pass
            else:
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ["wavelength", "intensity"]
                
                # Use Plotly to create a line plot
                fig = px.line(df, x="wavelength", y="intensity")
                plots.append(fig)

        # Check if the plots list is empty
        if len(plots) == 0:
            st.write("No plots to display")
        else:
            # Use the make_subplots function to combine all of the plots into a single figure
            fig = subplots.make_subplots(rows=len(plots), cols=1)

            # Add each plot to the figure as a separate trace
            for plot in plots:
                fig.add_traces(plot.data)

            # Update the layout to adjust the size of the plot
            fig.update_layout(width=800, height=3200)
            st.plotly_chart(fig)
            
if rad == "Analysis": 
    st.title("Analysis")
    # Spectrum 
    smootht1y_max = np.zeros(len(timeaxis))
    index = np.zeros(len(timeaxis))
    b = 0

    for idx, uploaded_file in enumerate(uploaded_files):
        if idx < starttime: pass
        elif idx > endtime: break
        elif (idx-1) % timeinterval != 0: pass
        else:
            df = pd.read_csv(uploaded_file, header=None)
            df.columns = ["wavelength", "intensity"]
            t1x = df["wavelength"]
            t1y = df["intensity"]

            filtert1y = df.loc[(df['wavelength'] < maxwavelength) & (df['wavelength'] > minwavelength)]
            filtert1y = filtert1y['intensity'].to_numpy().flatten()
            filtert1x = np.linspace(minwavelength, maxwavelength, max(filtert1y.shape))

            smootht1y = smooth(filtert1y,ns)

            interpt1y = interp1d(filtert1x,smootht1y)
            interpt1x = np.linspace(minwavelength, maxwavelength, round(max(filtert1y.shape)*ir*100))

            smootht1y_max[b] = max(interpt1y(interpt1x))
            index[b] = np.argmax(interpt1y(interpt1x))
            b = b+1
            plt.plot(interpt1y(interpt1x))

            peaks, _ = find_peaks(interpt1y(interpt1x), distance=1)
            np.diff(peaks)
            plt.plot(peaks, interpt1y(interpt1x)[peaks], "x", color="red")    
    st.pyplot()

if rad == "Result":
    st.balloons()
    st.title("Result")
    smootht1y_max = np.zeros(len(timeaxis))
    index = np.zeros(len(timeaxis))
    b = 0

    for idx, uploaded_file in enumerate(uploaded_files):
        if idx < starttime: pass
        elif idx > endtime: break
        elif (idx-1) % timeinterval != 0: pass
        else:
            df = pd.read_csv(uploaded_file, header=None)
            df.columns = ["wavelength", "intensity"]
            t1x = df["wavelength"]
            t1y = df["intensity"]

            filtert1y = df.loc[(df['wavelength'] < maxwavelength) & (df['wavelength'] > minwavelength)]
            filtert1y = filtert1y['intensity'].to_numpy().flatten()
            filtert1x = np.linspace(minwavelength, maxwavelength, max(filtert1y.shape))

            smootht1y = smooth(filtert1y,ns)

            interpt1y = interp1d(filtert1x,smootht1y)
            interpt1x = np.linspace(minwavelength, maxwavelength, round(max(filtert1y.shape)*ir*100))

            smootht1y_max[b] = max(interpt1y(interpt1x))
            index[b] = np.argmax(interpt1y(interpt1x))
            b = b+1

            peaks, _ = find_peaks(interpt1y(interpt1x), distance=1)
            np.diff(peaks)

    plt.subplot(1,2,1)
    plt.plot(timeaxis, index, '.')
    plt.subplot(1,2,2)
    plt.plot(timeaxis, smootht1y_max, '.')
    st.pyplot()

