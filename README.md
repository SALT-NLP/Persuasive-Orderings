# Persuasive Orderings
This repo contains code for the following paper: 

*Omar Shaikh, Jiaao Chen, Jon Saad-Falcon, Duen Horng (Polo) Chau, Diyi Yang*: Examining the Ordering of Rhetorical Strategies in Persuasive Requests  (EMNLP (Findings) 2020)

To see an overview of our analyses, take a look at pattern_finding.ipynb. 

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* transformers
* Pandas, Numpy, Pickle

### Code Structure
```
|__code/
        |__ vae_train/ --> folder for our VAE model. Run train.py to train this model.
        |__ dataset_iterators.py --> specific iterators for different analyses.
        |__ pattern_finding.ipynb --> step by step breakdown of the analysis in the paper.
        |__ baselines.ipynb --> notebook for evaluating baselines.
        |__ lstm_train/ --> folder for our LSTM model. Import the train method from train.py, and follow steps in pattern_finding.ipynb to train this.
        |__ editing_utils.py/ --> utilities to handle edits made to requests
```

### Instructions

#### Training the VAE
Please run `train.py` in `code/vae_train/`

#### Analyzing + Training LSTM
Please run `./pattern_finding.ipynb`

#### Running Baselines
Please run `./baselines.ipynb` to train the BERT baseline model and Naive Bayes.
