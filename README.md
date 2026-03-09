# First_Motion_Polarity model

### A SeisBench implementation of First-Motion Polarity (FMP) Determination model by Ross, Meier and Hauksson 2018 (https://doi.org/10.1029/2017JB015251)
#### This repository includes:

1. "FMP_seisbench.ipynb" = Jupyter Notebook demonstrating how to run the FMP SeisBench implementation (FMP_seisbench) on an example seismic event
3. "original.pt" = pretrained model weights of the First Motion Polarity Determination model described by Ross et al. (2018), converted from the original implementation into a PyTorch-compatible format
4. "fmp_seisbench.py" = Python module containing the SeisBench implementation of the FMP model, including a new function that generates probability traces using sliding windows for "Up/Down/Unknown" classes and produces a polarity classification using the mean probability within sliding windows +- 0.25 seconds around the P arrival pick time 
5. "FMP_seisbench_requirements.txt" = text file containing Python package dependencies required to run the code
