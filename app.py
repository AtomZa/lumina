import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import plotly.express as px
import plotly.subplots as subplots
import plotly.graph_objects as go

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Setting
st.set_option('deprecation.showPyplotGlobalUse', False)

# Navigation
rad = st.sidebar.radio("Menu",["Preprocessing","Analysis","Result"])

# Upload Files
uploaded_files = st.sidebar.file_uploader("Choose a CSV file", accept_multiple_files=True)

# Create an empty dictionary to store the groupings
groups = {}

# Check if any files were uploaded
if uploaded_files:
    # Iterate over the list of uploaded files
    for file in uploaded_files:

        # Split the file name into parts
        name_parts = file.name.split('_')
        
        # Extract the name code from the file name
        name_code = name_parts[0]
        
        # If the name code is not already a key in the dictionary, add it as a key and create a new list for the group
        if name_code not in groups:
            groups[name_code] = []
        
        # Add the file to the group with the corresponding name code
        groups[name_code].append(file)

    # Print the groups
    selected_key = st.sidebar.selectbox("Select a key", options=list(groups.keys()))

# Create a file uploader widget
ref_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

trans = False
# If a file was uploaded
if ref_file:
    ref = pd.read_csv(ref_file, header=None)

    # Display a checkbox
    trans = st.sidebar.checkbox("Transmittance")

# Parameter 1
st.sidebar.subheader("Smooth & Interpolate")
ns = st.sidebar.number_input("Smooth Factor", min_value = 0, value = 100, step = 10)
ir = st.sidebar.number_input("Interpolate Interval", min_value = 0.000, value = 0.001, step = 0.001, format="%.3f")

# Parameter 2
st.sidebar.subheader("Time Interval")
c, d, e = st.columns(3)
starttime = st.sidebar.number_input("Start Time", min_value = 0, value = 1, step = 1)
timeinterval = st.sidebar.number_input("Time Interval", min_value = 0, value = 5, step = 1)
endtime = st.sidebar.number_input("End Time", min_value = 0, value = 16, step = 1)
timeaxis = np.arange(starttime, endtime+timeinterval, timeinterval)

# Parameter 3
st.sidebar.subheader("Wavelength")
minwavelength = st.sidebar.number_input("Min Wavelength", min_value = 0, value = 200, step = 10)
maxwavelength = st.sidebar.number_input("Max Wavelength", min_value = 0, value = 1000, step = 10)

if rad == "Preprocessing":
    if uploaded_files:

        # Create dropdown menu for values based on selected key
        value_select = st.selectbox("Select a value:", groups[selected_key])

        val = pd.read_csv(value_select, header=None)
        val.columns = ["wavelength", "intensity"]
        t1x = val["wavelength"]
        t1y = val["intensity"]

        # Create a streaming line chart using Plotly Express
        fig1 = px.line(val, x=t1x, y=t1y)

        filtert1y = val.loc[(val['wavelength'] < maxwavelength) & (val['wavelength'] > minwavelength)]
        filtert1y = filtert1y['intensity'].to_numpy().flatten()
        filtert1x = np.linspace(minwavelength, maxwavelength, max(filtert1y.shape))

        smootht1y = smooth(filtert1y,ns)

        interpt1y = interp1d(filtert1x,smootht1y)
        interpt1x = np.linspace(minwavelength, maxwavelength, round(max(filtert1y.shape)*ir*100))

        # Use the px.line function to create a Plotly line plot
        fig2 = px.line(x=interpt1x, y=interpt1y(interpt1x))

        # Combine the two figures into a single Plotly figure object
        fig = fig1.add_trace(fig2.data[0])

        # Add the combined figure to the Streamlit app
        st.plotly_chart(fig)
            
if rad == "Analysis": 
    # Create an empty list to store the plots
    plots = []
    
    # Spectrum 
    smootht1y_max = np.zeros(len(timeaxis))
    index = np.zeros(len(timeaxis))
    b = 0

    for idx, uploaded_file in enumerate(groups[selected_key], start=1):
    #for idx, uploaded_file in enumerate(uploaded_files):
        if idx < starttime:
            pass
        elif idx > endtime:
            break
        elif (idx-1) % timeinterval != 0:
            pass
        else:
            if trans :
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ["wavelength", "intensity"]
                t1x = df["wavelength"]
                t1y = df["intensity"]

                ref.columns = ["wavelength", "intensity"]
                ref_x = ref["wavelength"]
                ref_y = ref["intensity"]

                filtert1y = df.loc[(df['wavelength'] < maxwavelength) & (df['wavelength'] > minwavelength)]
                filtert1y = filtert1y['intensity'].to_numpy().flatten()
                filtert1x = np.linspace(minwavelength, maxwavelength, max(filtert1y.shape))

                filterrefy = ref.loc[(ref['wavelength'] < maxwavelength) & (ref['wavelength'] > minwavelength)]
                filterrefy = filterrefy['intensity'].to_numpy().flatten()
                filterrefx = np.linspace(minwavelength, maxwavelength, max(filterrefy.shape))

                smootht1y = smooth(filtert1y,ns)

                smoothtref = smooth(filterrefy,ns)

                interpt1y = interp1d(filtert1x,smootht1y)
                interpt1x = np.linspace(minwavelength, maxwavelength, round(max(filtert1y.shape)*ir*100))

                interprefy = interp1d(filterrefx,smoothtref)
                interprefx = np.linspace(minwavelength, maxwavelength, round(max(filterrefy.shape)*ir*100))

                smootht1y_max[b] = max(interpt1y(interpt1x)/interprefy(interprefx))
                index[b] = np.argmax(interpt1y(interpt1x)/interprefy(interprefx))
                
                b = b+1

                # Use the px.line function to create a Plotly line plot
                fig = px.line(x=interpt1x, y=interpt1y(interpt1x)/interprefy(interprefx))

                # Add the peaks to the plot
                peaks, _ = find_peaks(interpt1y(interpt1x)/interprefy(interprefx), distance=1)
                fig.add_scatter(x=interpt1x[peaks], y=(interpt1y(interpt1x)/interprefy(interprefx))[peaks], mode="markers", marker=dict(color="red"), name=f"Spectrum {idx}")

                # Add the plot to the list of plots
                plots.append(fig)    
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

                # Use the px.line function to create a Plotly line plot
                fig = px.line(x=interpt1x, y=interpt1y(interpt1x))

                # Add the peaks to the plot
                peaks, _ = find_peaks(interpt1y(interpt1x), distance=1)
                fig.add_scatter(x=interpt1x[peaks], y=interpt1y(interpt1x)[peaks], mode="markers", marker=dict(color="red"), name=f"Spectrum {idx}")

                # Add the plot to the list of plots
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

if rad == "Result":
    st.balloons()
    st.title("Result")

    # Create a subplot figure with two columns
    fig = subplots.make_subplots(rows=1, cols=2)

    sensitive = np.zeros(len(groups))
    a = 0
    
    for key, values in groups.items():

        smootht1y_max = np.zeros(len(timeaxis))
        index = np.zeros(len(timeaxis))
        b = 0

        for idx, uploaded_file in enumerate(values, start=1):
            if idx < starttime: pass
            elif idx > endtime: break
            elif (idx-1) % timeinterval != 0: pass
            else:
                if trans:
                    df = pd.read_csv(uploaded_file, header=None)
                    df.columns = ["wavelength", "intensity"]
                    t1x = df["wavelength"]
                    t1y = df["intensity"]

                    ref.columns = ["wavelength", "intensity"]
                    ref_x = ref["wavelength"]
                    ref_y = ref["intensity"]

                    filtert1y = df.loc[(df['wavelength'] < maxwavelength) & (df['wavelength'] > minwavelength)]
                    filtert1y = filtert1y['intensity'].to_numpy().flatten()
                    filtert1x = np.linspace(minwavelength, maxwavelength, max(filtert1y.shape))

                    filterrefy = ref.loc[(ref['wavelength'] < maxwavelength) & (ref['wavelength'] > minwavelength)]
                    filterrefy = filterrefy['intensity'].to_numpy().flatten()
                    filterrefx = np.linspace(minwavelength, maxwavelength, max(filterrefy.shape))

                    smootht1y = smooth(filtert1y,ns)

                    smoothtref = smooth(filterrefy,ns)

                    interpt1y = interp1d(filtert1x,smootht1y)
                    interpt1x = np.linspace(minwavelength, maxwavelength, round(max(filtert1y.shape)*ir*100))

                    interprefy = interp1d(filterrefx,smoothtref)
                    interprefx = np.linspace(minwavelength, maxwavelength, round(max(filterrefy.shape)*ir*100))

                    smootht1y_max[b] = max(interpt1y(interpt1x)/interprefy(interprefx))
                    index[b] = np.argmax(interpt1y(interpt1x)/interprefy(interprefx))

                    if(idx == 1):
                        sensitive[a] = index[b]

                    b = b+1
                    
                    peaks, _ = find_peaks(interpt1y(interpt1x)/interprefy(interprefx), distance=1)

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

                    if(idx == 1):
                        sensitive[a] = index[b]
                    
                    b = b+1

                    peaks, _ = find_peaks(interpt1y(interpt1x), distance=1)
        
        a = a+1

        # Add the first plot to the figure
        fig.add_scatter(x=timeaxis, y=index, mode="markers", row=1, col=1, name="Wavelength " + str(key))

        # Add the second plot to the figure
        fig.add_scatter(x=timeaxis, y=smootht1y_max, mode="markers", row=1, col=2, name="Intensity " + str(key))

    # Add the scatter plot to the Streamlit app
    st.plotly_chart(fig)

    # Define the x and y values
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]

    # Create the scatter plot
    fig = go.Figure(data=go.Scatter(x=x, y=y))

    # Add the scatter plot to the Streamlit app
    st.plotly_chart(fig)

    # Create a copy of the original data
    x_filtered = x.copy()
    y_filtered = y.copy()

    # Create the scatter plot
    fig = go.Figure(data=go.Scatter(x=x_filtered, y=y_filtered))

    # Add an event listener that removes the data point when it is clicked
    # Remove the customdata_custom property
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color="red"),
            hoverinfo="none",
            showlegend=False,
            visible=False,
            customdata=list(range(len(x))),  # Set the customdata property directly
            clickmode="event+select"
        )
    )

    # Set the layout of the plot to update the data when a point is clicked
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        label="Remove data point",
                        method="restyle",
                        args=[{"visible": [False, True]}, {"title": "Data point removed"}]
                    )
                ]),
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.57,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )

    # Add the scatter plot to the Streamlit app
    st.plotly_chart(fig)

    # Add a callback function that updates the filtered data when a point is clicked
    @st.cache(persist=True)
    def remove_point(data):
        # Get the index of the point to remove
        index = data["points"][0]["customdata"][0]
        # Remove the point from the filtered data
        x_filtered.pop(index)
        y_filtered.pop(index)
        # Return the updated data
        return x_filtered, y_filtered

    # Add the callback function to the scatter plot
    fig.data[0].on_click(remove_point)
