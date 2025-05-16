# GGANDTA

Generative Adversarial Network for Predicting Drug Target Binding Affinity. It is
built with **Pytorch* and **Python 3**.


## Installation

### Requirements
  * python 3.9, numpy, scipy, pandas, pytorch, pyg

#### 1. Create a virtual environment

```bash
# create
conda create -n GGANDTA python=3.9
# activate
conda activate GGANDTA
# deactivate
conda deactivate
```

#### 2. clone GGANDTA
- After creating and activating the GGANDTA virtual environment, download GGANDTA from github:
```bash
git clone https://github.com/wz/GGANDTA.git
cd GGANDTA
```
#### 3. Install

```bash
conda activate GGANDTA
conda install numpy, scipy, pandas, Pytorch, pyg

```



## Tested data
The example data can be downloaded from 
#### Davis and KIBA
https://github.com/thinng/GraphDTA/tree/master/data



## Usage

### Train Model

#### 1. Create Dataset

```bash
python data_creation.py

```
First, divide the data into training and test sets and create data files in pytorch format.
#### 2. Train model

Run the following script to train the model.
```bash
python train.py

```



#### 3. Validate the training prediction model

Run the following script to test the model.
```bash
python valid.py

```
This returns the best MSE model for the validation dataset during the training process.


