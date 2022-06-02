# Deep-learning-reconstruction-Radial-SMS-perfusion 

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

<<<<<<< HEAD
<<<<<<< HEAD
The T1NET was trained on a Quadro RTX 6000 GPU (~24 GB) on a Linux Fedora 26 operatoring system. Trainiing the network for 100 epochs requires ~18 hours.
=======
The T1NET was trained on a Quadro RTX 6000 GPU (~24 GB) on a Linux Fedora 26 operatoring system. Training the network for 100 epochs requires ~18 hours.
>>>>>>> 3c76f6d132d261b870c8644a7534c113da35c6dd
=======
The T1NET was trained on a Quadro RTX 6000 GPU (~24 GB) on a Linux Fedora 26 operatoring system. Training the network for 100 epochs requires ~18 hours.
>>>>>>> 3c76f6d132d261b870c8644a7534c113da35c6dd



https://user-images.githubusercontent.com/35586452/129275208-073007a2-6466-48d0-b807-616cae605c15.mp4



|           |        BU2          |        MoDL         |     CRNN-MRI        |        BU3          |    BU3 (ungated)    |
|:---------:|:-----------:|:-----------:|:------------:|:-----------:|:-----------:|
|   SSIM    |   0.807 ± 0.034     |   0.720 ± 0.036     |   0.935 ± 0.029     |   0.963 ± 0.012     |   0.915 ± 0.028     |
|   PSNR    |   32.084 ± 1.960    |   30.145 ± 1.408    |   38.707 ± 2.850    |   40.238 ± 2.424    |   35.239 ± 2.670    |
|   NRMSE   |   0.375 ± 0.069     |   0.468 ± 0.075     |   0.149 ± 0.052     |   0.147 ± 0.033     |   0.181 ± 0.037     |
|   TIME (s)|         7           |         98          |        110          |         8           |         12          |

Performance comparisons for T1 map generation using the reduced model, a muti-layer perceptrion (MLP), U-Net, the T1NET, nd reference T1 maps. Numbers correspond to averages for AHA regions of interest in the myocardium (MYO) nd the left-ventricular blood pool (LVB) with their corresponding stndard deviations, Mean ± SD. P-values indicate the statistical significance of network and reduced three-parameter model T1 maps in comparison to the reference T1 maps with p < 0.05.

Contact: 

Johnathan Le

le.johnv@outlook.com

Ganesh Adluru

gadluru@gmail.com
