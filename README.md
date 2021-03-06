# Accelerated cardiac T1 mapping with recurrent networks and cyclic, model-based loss

This repository contains code and example test datasets for the paper 'Accelerated cardiac T1 mapping with recurrent networks and cyclic, model-based loss'

Instructions:
Running this code requires the installation of Conda and Pip for creating virtual environments and installing packages. Instructions for installing conda and pip can be found in the following link:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

https://pip.pypa.io/en/stable/installation/

This repository only contains example test datasets for visualizing the results of the T1NET paper. Additional  datasets can be provided upon request.

1. Create a virtual environment using the command 'conda create -n myenv python=3.6.7'
2. Activate the virtual environment using the command 'source activate myenv'
3. Run 'pip install -r requirements.txt'
4. Run 'train_network.py in the precontrast or postcontrast directories to train the respective networks, trained networks save to the trainedNetwork directory
5. Run 'visualize.py' to visualize a test dataset, movies of the reconstructions saves to the results file

The T1NET was trained on a Quadro RTX 6000 GPU (~24 GB) on a Linux Fedora 26 operating system. Training the network for 100 epochs requires ~18 hours.

<br />
<br />
<br />

![Figure2](https://user-images.githubusercontent.com/35586452/171719902-e8a9514c-d09e-460f-bc8a-90c2a818094a.png)
Figure 1. Myocardial pre-contrast T1 mapping results of T1NET and the reduced model fitting (T=3). (A) Correlation and Bland-Altman plots of the T1NET. Each dot corresponds to the average of a myocardial AHA segment.  (B) Correlation and Bland-Altman plots of the reduced model fitting. (C) Network-generated pre-contrast short-axis cardiac T1 maps in comparison to reference T1 maps and their corresponding difference images. (D) Reduced model-generated pre-contrast short-axis cardiac T1 maps in comparison to reference T1 maps and their corresponding difference images. 

<br />
<br />
<br />


<p align="center">
<img width="488" alt="Table" src="https://user-images.githubusercontent.com/35586452/171721210-8d18ff39-df9c-4535-b9e8-8937fe136824.png">
</p>
Table 1. Performance comparisons for T1 map generation using the reduced model, a multi-layer perceptron (MLP), U-Net, the T1NET, and reference T1 maps. Numbers correspond to averages in milliseconds for AHA regions of interest in the myocardium (MYO) and regions of interest in the left ventricular blood pool (LVBP) with their corresponding standard deviations, Mean ?? SD. P-values indicate the statistical significance of network and reduced three-parameter model T1 maps in comparison to the reference T1 maps with p<0.05.

<br />
<br />
<br />

Contact: 

Johnathan Le

le.johnv@outlook.com

Ganesh Adluru

gadluru@gmail.com
