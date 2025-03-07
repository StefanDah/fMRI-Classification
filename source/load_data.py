import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image, datasets

# %%
def load_data(data_path):   
    """
    Load subject data and labels.

    Parameters:
    data_path (str): Path to the directory containing subject data.

    Returns:
    tuple: A tuple containing:
        - subjects (list): List of subject IDs.
        - subject_labels (dict): Dictionary mapping subject IDs to binary group labels.
        - data_path (str): Path to the directory containing subject data.
    """
    # Load subject labels from a TSV file (format: subject_id\tgroup)
    labels_df = pd.read_csv("data/participants.tsv", sep="\t")  
    labels_df['group_binary'] = labels_df['group'].map({'depr': 1, 'control': 0})
    subject_labels = dict(zip(labels_df["participant_id"], labels_df["group_binary"]))  # Convert to dictionary

    # List all subjects
    subjects = [sub for sub in os.listdir(data_path) if sub.startswith("sub-")]
    print(f"Found {len(subjects)} subjects.")

    return(subjects,subject_labels, data_path)


# %%
def preprocessing(subjects, data_path):    
    """
    Preprocess fMRI data for each subject.

    Parameters:
    subjects (list): List of subject IDs.
    data_path (str): Path to the directory containing subject data.

    Returns:
    None
    """
    # Load the standard MNI152 template (2mm resolution)
    mni_template = datasets.load_mni152_template(resolution=2)

    for subject in subjects:
        try:
            print(f"Processing {subject}...")
            subject_path = os.path.join(data_path, subject)
            
            # Load subject's T1-weighted MRI
            anat_file = os.path.join(subject_path, "anat", f"{subject}_T1w.nii.gz")
            t1_img = nib.load(anat_file)
            
            # Load subject's fMRI scan
            fmri_file = os.path.join(subject_path, "func", f"{subject}_task-rest_bold.nii.gz")
            fmri_img = nib.load(fmri_file)
            
            # Co-registration of fMRI and T1-weighted scans
            registered_fmri_img = image.resample_to_img(fmri_img, t1_img, interpolation='linear')
            print(f"{subject} co-registration successful")  

            # Resample the full 4D fMRI image to MNI space
            fmri_mni = image.resample_img(
                registered_fmri_img, target_affine=mni_template.affine, target_shape=mni_template.shape, 
                interpolation="continuous", force_resample=True, copy_header=True
            )
            print(f"{subject} resampled to MNI space")  

            # Smooth the MNI resampled fMRI image with a 6mm gaussian filter
            smoothed_fmri_img = image.smooth_img(fmri_mni, fwhm=6)
            smoothed_file = os.path.join(subject_path, "func", f"{subject}_task-rest_bold_MNI_smooth.nii.gz")
            # Save smoothed images 
            smoothed_fmri_img.to_filename(smoothed_file)
            
            print(f"{subject} MNI fMRI smoothed and saved.")
            
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            
    return