# SCL-Fish


This repo contains the codes to reproduce our work. We will make all the codes publicly available upon acceptance.


## 1. Requirements:

Please run the following commands:</br>
```diff
pip install numpy
pip install pandas
pip install torch
pip install sklearn
pip install tqdm
pip install transformers
pip install datasets
pip install nltk
pip install spacy
pip install matplotlib
```


## 2. Data Preprocessing:

Please collect the datasets that we used in our paper from their corresponding original works.</br>
After downloading, please put the datasets in ```Datasets``` directory.</br>

Run the following commands:
```diff
python3 Data_Processing_Fish.py
python3 Data_Processing_ERM.py
```

The preprocessed datasets will be stored in ```Pickles/Domain``` directory.</br>


## 3. Run Model:

To run the models, please execute the following command:
```diff
python3 [MODEL_NAME].py
```
MODEL_NAME = SCL_fish/fish/ERM/SCL_ERM

To facilitate the review process, the models are already set to their respective hyperparameters used in the experiments.
We ran the experiments on Nvidia A100 40GB GPU.



## 4. Evaluation:

To evaluate on the cross-platform datasets, please run the following command:
```diff
python3 Eval.py
```
