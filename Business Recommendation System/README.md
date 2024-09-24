Business Recommendation Systems
==============================

Personalized recommendation system tailored to historical preferences of yelp users.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

## Overview
This project implements a recommendation system using the Yelp dataset, combining collaborative filtering and content-based filtering methods. The model aims to predict user ratings for businesses based on user and business features, utilizing PyTorch for deep learning.

## Dataset
The dataset used for this project is derived from Yelp, focusing on user and business information. Key preprocessing steps include:

- Removal of users with fewer than 5 reviews.
- Removal of businesses with fewer than 5 reviews.

## Features
The input features for the model are constructed from the following components:

- **User Features**: 
  - User ID (`user_num_id`)
  - User PCA features (`u_pca1`, `u_pca2`, `u_pca3`)
  
- **Business Features**: 
  - Business ID (`business_num_id`)
  - Business average rating normalized
  
- **Contextual Features**: 
  - Region ID (`region_code`)
  - Day of the week (`day_of_week`)
  - Day of the year (`day_of_year`)
  
- **Tokenized Text Features**: 
  - Reviews tokenized using the BERT tokenizer, along with the corresponding attention masks.

## Model Architecture
The model utilizes an embedding-based architecture for collaborative filtering combined with a neural network for content-based features:

- User and business embeddings are learned using `nn.Embedding`.
- A fully connected layer processes PCA features.
- The outputs of these layers are concatenated and passed through additional fully connected layers to predict user ratings.

## Training
The model is trained using the following configuration:

- **Optimizer**: AdamW, which incorporates weight decay regularization.
- **Loss Function**: Mean Squared Error (MSE) for regression tasks.
- **Batch Size**: 1000
- **Learning Rate**: 0.001
- **Number of Epochs**: 20

The training loop includes:

- Logging of training and validation losses.
- Gradient norm logging for monitoring model training.

## Data Handling
Data is split into training, validation, and test sets using the `train_test_split` function from Scikit-learn. The datasets are saved in Parquet format for efficient loading.

### Code Structure
The primary components of the code are as follows:

1. **Dataset Class**: 
   - `CFDataset` handles the input data preparation, including feature extraction and target labeling.

2. **Data Splitting and Saving**:
   - `split_and_save_data`: Splits the dataset into training, validation, and test sets, saving them along with the tokenized tensors.

3. **Model Definition**: 
   - `CFmodel`: Defines the neural network architecture, including embeddings and fully connected layers.

4. **Training Loop**: 
   - Handles the training process, logging metrics, and saving the trained model.

## Getting Started
To run this project, ensure you have the required dependencies installed:

```bash
pip install torch numpy pandas scikit-learn


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
