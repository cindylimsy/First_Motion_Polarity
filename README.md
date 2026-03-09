# First_Motion_Polarity model

### A Seisbench implementation of First-Motion Polarity Determination model by Ross, Meier and Hauksson 2018 (https://doi.org/10.1029/2017JB015251)
#### This repository includes:

1. "FMP_seisbench.ipynb" = Jupyter Notebook running the FMP_seisbench code on an example 
2. "original.pt" = model weights from the First Motion Polarity Determine model (Ross et al. 2018) translated into PyTorch
3. "fmp_seisbench.py" = code to run the model with new function to produce probability traces for "Up/Down/Unknown" classes and take the mean of the sliding windows
4. "FMP_seisbench_requirements.txt" = text file containing package dependencies for the Python environment
