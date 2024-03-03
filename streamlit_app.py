import streamlit as st
#from PyEMD import EMD
import scipy.io as spio
import numpy as np
import pywt

#from gensim.similarities import WmdSimilarity
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.signal import butter,lfilter
#import numpy as np
from scipy.stats import entropy
import pandas as pd
from pyentrp import entropy as ent

import matplotlib.pyplot as plt

def display_smooth_border_plot(x, y, xlabel='X-axis', ylabel='Y-axis', title='Plot'):
    # Create a plot using matplotlib
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.set_title(title)
    st.pyplot(fig)
    # Display the plot with smooth borders and interactive behavior
    st.markdown(
        f"""
        <div 
                       style="
                display: flex;
                justify-content: center;
                align-items: center;
                width: 50%; /* Adjust width as needed */
                border-radius: 10px;
                box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.2);
                transition: box-shadow 0.3s ease;
            "
            onmouseover="this.style.boxShadow='0px 0px 20px rgba(0, 0, 255, 0.5)';"
            onmouseout="this.style.boxShadow='0px 0px 20px rgba(0, 0, 0, 0.2)';"
        >
            {fig}
        </div>
        """,
        unsafe_allow_html=True
    )

# Load the pre-trained machine learning model
model = joblib.load("Linear_reg_iter2.pkl")

def sample_entropy(signal):
    std_ts = np.std(signal)
    sample_entropy = ent.sample_entropy(signal, 4, 0.2 * std_ts)
    return sample_entropy
def fuzzy_entropy(signal):
    fuzzy_entropy_value = ent.shannon_entropy(signal)
    return fuzzy_entropy_value

# def fractal_dimension(signal):
#     # Implement fractal dimension calculation
#     pass

# def fuzzy_entropy(signal):
#     # Implement fuzzy entropy calculation
#     pass


def box_count(data, epsilon):
    data_min = np.min(data)
    data_max = np.max(data)
    num_boxes = int((data_max - data_min) / epsilon)
    count = 0
    for i in range(num_boxes):
        box_start = data_min + i * epsilon
        box_end = box_start + epsilon
        box_data = data[(data >= box_start) & (data < box_end)]
        if len(box_data) > 0:
            count += 1
    return count

def fractal_dimension(data, epsilon_min=0.9, epsilon_max=5, num_epsilon=10):
    epsilon_values = np.linspace(epsilon_min, epsilon_max, num_epsilon)
    counts = []

    for epsilon in epsilon_values:
        count = box_count(data, epsilon)
        counts.append(count)

    # Filter out zero counts
    epsilon_values = epsilon_values[np.array(counts) > 0]
    counts = np.array(counts)[np.array(counts) > 0]

    if len(epsilon_values) == 0 or len(counts) == 0:
        return None

    # Fit a linear regression to estimate the fractal dimension
    coeffs = np.polyfit(np.log(epsilon_values), np.log(counts), 1)
    fractal_dim = -coeffs[0]

    return fractal_dim

def interquartile_range(signal):
    return np.percentile(signal, 75) - np.percentile(signal, 25)

def mean_absolute_deviation(signal):
    return np.mean(np.abs(signal - np.mean(signal)))

def mean_energy(signal):
    return np.mean(np.square(signal))

def mean_teager_kaiser_energy(signal):
    return np.mean(np.square(np.diff(signal)))

# def sample_entropy(signal):
#     # Implement sample entropy calculation
#     pass

def standard_deviation(signal):
    return np.std(signal)

def perform_dwt(signal, wavelet, level):
    # Perform DWT using pywt.wavedec
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

# Define a function to perform Wavelet Packet Decomposition (WPD)
def perform_wpd(signal, level):
    # Choose a wavelet (e.g., 'db1' for Daubechies wavelet)
    wavelet = 'db8'

    # Perform WPD using pywt.wavedec
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Select a subset of coefficients for further processing
    selected_coeffs = coeffs[1:]  # Exclude the approximation coefficients at level 0

    return selected_coeffs


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply the bandpass filter to a signal
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

t=np.linspace(0,35000/20,35000)
# Signal_data=None
# cut_signal=None
fs=20

# Define function to preprocess input data
def preprocess_data(file):
    if str(file.name).endswith('.mat'):
        # file_contents = io.BytesIO(file_path.read())
        data = spio.loadmat(file)
        
        s1=data['val'][1][:35000]
        s2=data['val'][3][:35000]
        s3=data['val'][5][:35000]
        Signal_data=(s1+s2+s3)/3
        
        
        display_smooth_border_plot(t,Signal_data,xlabel='Time(s)',ylabel='Amplitude',title="Raw Signal Data")

        #Filtering
        lowcut=0.3
        highcut=3
        order=4
        fs=20
        
        smoothed_signal=butter_bandpass_filter(Signal_data, lowcut, highcut, fs, order)
        cut_signal=smoothed_signal[(3*60*fs):-(3*60*fs)]
        #st.write(len(cut_signal))
        display_smooth_border_plot(t[(3*60*fs):-(3*60*fs)],cut_signal,xlabel='Time(s)',ylabel='Amplitude',title="Smoothed and cut Signal")

            
        Signal_feature_data=[]


        # Assuming 'signal' is the EHG signal
        signal = cut_signal # Example signal
        levels = 5   # Example number of decomposition levels
        wavelet='db4'

        # Initialize lists to store features
        features = []

        # Perform DWT on the signal
        coeffs = perform_dwt(signal, wavelet, level=levels)

        # Perform WPD on each IMF and extract features
        for imf in coeffs:
            # Perform WPD and select coefficients
            wpd_coeffs = []  # Store selected WPD coefficients here
            for level in range(levels):
                selected_coeffs = perform_wpd(imf, level)
                wpd_coeffs.extend(selected_coeffs)

            # Calculate features for each coefficient
            for coeff in wpd_coeffs:
                features.append([
                    fractal_dimension(coeff),   #                # Fractal Dimension
                    fuzzy_entropy(coeff),     #                 # Fuzzy Entropy
                    interquartile_range(coeff),                # Interquartile Range
                    mean_absolute_deviation(coeff),            # Mean Absolute Deviation
                    mean_energy(coeff),                        # Mean Energy
                    mean_teager_kaiser_energy(coeff),          # Mean Teager-Kaiser Energy
                    #sample_entropy(coeff),  #                   # Sample Entropy
                    standard_deviation(coeff),                 # Standard Deviation
                ])
        Signal_feature_data.append(features)
        #st.write(len(Signal_feature_data[0]))
        dset=[]
        temp_df=pd.DataFrame(Signal_feature_data[0]).describe()
        #st.write(temp_df.shape)
        #st.write(temp_df.columns)
        dset.append(np.concatenate((temp_df[0].values,temp_df[1].values,temp_df[2].values,temp_df[3].values,temp_df[4].values,temp_df[5].values,temp_df[6].values),axis=0))
        processed_data=dset

        #processed_data=Signal_feature_data
        #st.write(len(Signal_feature_data))
        # Feature_arr=np.array(Signal_feature_data)
        # st.write(f"Feature_arr Shape: {Feature_arr.shape}")
        # Feature_arr_reshape=Feature_arr.flatten()
        # #Final_dataset_=Feature_arr_reshape   
        # st.write(f"Feature_arr_reshape Shape: {Feature_arr_reshape.shape}")

        # df=pd.Series(Feature_arr_reshape)
        # df.dropna(inplace=True)
        # processed_data=df.values
        # st.write(f"Processed_Data Shape: {processed_data.shape}")

 # Extract data from the .mat file
            # Adjust preprocessing according to your data structure
        return processed_data

def main():
    st.title("EHG Delivery Prediction")

    # Add CSS styles
    #st.title("Data Preprocessing and Visualization")
    st.markdown(
        """
        <style>
        body {
            background-color: #2C3E50;
            color: #ECF0F1;
            text-align: center;
        }
        .title {
            color: #3498DB;
            padding-top: 20px;
            padding-bottom: 20px;
            font-size: 24px;
        }
        .btn-primary {
            background-color: #2980B9;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)



    # File upload widget
    uploaded_file = st.file_uploader("Upload .mat file", type=['mat'])

    #st.write("Uploaded file contents:")
    #st.write(uploaded_file.getvalue())

    if uploaded_file is not None:
        # Perform prediction once file is uploaded and button is clicked

        if st.button("Predict"):
            # Preprocess the uploaded file
            data = preprocess_data(uploaded_file)
            #data=data.reshape((1,-1))

            # Make prediction
            prediction = model.predict(data)<=0.7
            st.write(f"Prediction:{prediction} " )
            # Display prediction result
            if np.round(prediction) ==1:
                st.success("Prediction: Term Delivery")
            else:
                st.warning("Prediction: Preterm Delivery")

            
            # Sample data
            # x = np.linspace(0, 10, 100)
            # y = np.sin(x)

            # Call the function to display the plot
                    

if __name__ == "__main__":
    main()
