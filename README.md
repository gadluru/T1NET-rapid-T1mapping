# Accelerated cardiac T1 mapping with recurrent networks and cyclic, model-based loss

This repository contains code and example test datasets for the paper 'Accelerated cardic T1 mapping with recurrent networks and cyclic, model-based loss'

Instructions:
Running this code requires the installation of Conda and Pip for creating virtual environments and installing packages. Instructions for installing conda and pip can be found in the following link:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

https://pip.pypa.io/en/stable/installation/

This repository only contains example test datasets for visualizing the results of T1NET paper. Additionl  datasets can be provided upon request.

1. Create a virtual environment using the command 'conda create -n myenv python=3.6.7'
2. Activate the virtual environment using the command 'source activate myenv'
3. Run 'pip install -r requirements.txt'
4. Run 'train_network.py in the precontrast or postcontrast directories to train the respective networks, traind networks save to the trainedNetwork directory
5. Run 'visualize.py' to visualize a test dataset, movies of the reconstructions saves to the results file

The T1NET was trained on a Quadro RTX 6000 GPU (~24 GB) on a Linux Fedora 26 operatoring system. Trainiing the network for 100 epochs requires ~18 hours.


![Figure2](https://user-images.githubusercontent.com/35586452/171718146-750d010a-04c4-46aa-9f31-cc96f1e2fbaf.png)
Figure 1. Myocardial pre-contrast T1 mapping results of T1NET and the reduced model fitting (T=3). (A) Correlation and Bland-Altman plots of the T1NET. Each dot corresponds to the average of a myocardial AHA segment.  (B) Correlation and Bland-Altman plots of the reduced model fitting. (C) Network-generated pre-contrast short-axis cardiac T1 maps in comparison to reference T1 maps and their corresponding difference images. (D) Reduced model-generated pre-contrast short-axis cardiac T1 maps in comparison to reference T1 maps and their corresponding difference images. 




<img width="468" alt="image" src="https://user-images.githubusercontent.com/35586452/171719414-1a78c454-0d18-4be7-a28d-a96b89574776.png">

Table 1. Performance comparisons for T1 map generation using the reduced model, a multi-layer perceptron (MLP), U-Net, the T1NET, and reference T1 maps. Numbers correspond to averages for AHA regions of interest in the myocardium (MYO) and regions of interest in the left ventricular blood pool (LVBP) with their corresponding standard deviations, mean±SD. P-values indicate the statistical significance of network and reduced three-parameter model T1 maps in comparison to the reference T1 maps with p<0.05.

Contact: 

Johnathan Le

le.johnv@outlook.com

Ganesh Adluru

gadluru@gmail.com
