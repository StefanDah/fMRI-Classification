# %%
import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image, datasets, connectome
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure


# %%
def compute_roi_time_series(subjects, data_path):
    """
    Compute ROI-based time series for each subject.

    Parameters:
    subjects (list): List of subject IDs.
    data_path (str): Path to the directory containing subject data.

    Returns:
    np.ndarray: Array containing ROI-based time series for each subject.
    """
    #Create functional connectivity matrix
    #Extract ROI-based Time-Series
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=2)
    labels_masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=False, memory='nilearn_cache', n_jobs=-1)
    
    X_roi = []
    for subject in subjects:
        # Load smooothed scans
        subject_path = os.path.join(data_path, subject)
        smoothed_file = os.path.join(subject_path, "func", f"{subject}_task-rest_bold_MNI.nii.gz")
        smoothed_fmri_img = nib.load(smoothed_file)
        
        # Generate roi time series
        roi_time_series = labels_masker.fit_transform(smoothed_fmri_img)
        X_roi.append(roi_time_series)  # Store in list
        print(f"{subject} roi_time_series created")
        
        
    print(f"Number of unique subjects: {len(set(subjects))}")
    
    X_roi = np.array(X_roi)
    return X_roi

# %%
def compute_correlation_matrix(X_roi, subjects):
    """
    Compute the correlation matrix for each subject's ROI-based time series.

    Parameters:
    X_roi (np.ndarray): Array containing ROI-based time series for each subject.
    subjects (list): List of subject IDs.

    Returns:
    np.ndarray: Array containing the flattened correlation matrices for each subject.
    """
    X_correlation = []
    # Compute Functional Connectivity Matrix
    for subject in range(len(subjects)):
        correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
        covariance = correlation_measure.fit_transform([X_roi[subject]])[0]
        
        # Ensure the diagonal values are not zero to prevent divide-by-zero errors
        np.fill_diagonal(covariance, 1e-5)
        
        # Compute correlation safely
        correlation_matrix = covariance / np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        np.fill_diagonal(correlation_matrix, 0)
        
        # Flatten connectivity matrix into a feature vector
        X_correlation.append(correlation_matrix.flatten())
        
    X_correlation = np.array(X_correlation)
   
    return X_correlation


