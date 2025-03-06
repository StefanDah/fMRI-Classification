## Depression Classification based on fMRI scans

In this project i try to use different ML algorithms to classify healthy controls from patients suffering from mild depression using fMRI scans.  
Two different ML  algorithms are used:  
- SVC (Support vector machine)
- meanMLP (multi-layer perceptron based on https://github.com/neuroneural/meanMLP)  

meanMLP was developed by [Popov et al. (2024)](https://www.sciencedirect.com/science/article/pii/S1053811924004063#b19) specifically for fMRI time-series classification.


### Data source
The dataset used for this project is publicly available on OpenNeuro:  
- [OpenNeuro ds002748, version 1.0.5](https://openneuro.org/datasets/ds002748/versions/1.0.5)

### Preprocessing
- Importing the data and first visual inspection 
- Co-registration of fMRI scans with corresponding T1- weighted scans to ensure anatomical consistency
- Transform the data into the standard MNI space to facilitate comparison across subjects
- Apply 6mm Gaussian Smoothing filter

### Feature exctration
- ROI time series are generated using the Schaefer 200 atlas, which partitions the brain into 200 regions
- Only for SVC a functional connectivity matrix was generated (not necessary for meanMLP)

### Classification Model
- SVC model is trained on the connectivity matrix
- meanMLP is trained directly on the ROI time series

### Evaluation 
- Model performance is evaluated using ROC AUC

---

If you want to use this project follow these steps:

1. Clone the repository  
2. Set up virtual environment using requirements.txt
3. Download the data from [OpenNeuro ds002748, version 1.0.5](https://openneuro.org/datasets/ds002748/versions/1.0.5) and place into data/ 
4. Run main.py 
    ```bash
    python main.py
    ```
5. Evaluation plots are saved to /figures

---
### To-DO
First results seem promising but a lot of work needs to done on preprocessing and model optimization.
SVC is not fully implemented  yet.

---
### References

Bezmaternykh DD and Melnikov ME and Savelov AA and Petrovskii ED (2021). Resting state with closed eyes for patients with depression and healthy participants. OpenNeuro. [Dataset] [doi: 10.18112/openneuro.ds002748.v1.0.5](https://onlinelibrary.wiley.com/doi/10.

Pavel Popov, Usman Mahmood, Zening Fu, Carl Yang, Vince Calhoun, Sergey Plis,
A simple but tough-to-beat baseline for fMRI time-series classification,
NeuroImage 2024, https://doi.org/10.1016/j.neuroimage.2024.120909

meanMLP: https://github.com/neuroneural/meanMLP
