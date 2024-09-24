# Iris Flower Classification Using Logistic Regression

This project implements a logistic regression model to classify the Iris flower species based on their sepal and petal measurements. The dataset used for this analysis is the famous Iris dataset.

## Libraries Used

- `numpy`: For numerical operations
- `pandas`: For data manipulation and analysis
- `matplotlib`: For data visualization
- `seaborn`: For enhanced visualization
- `sklearn`: For machine learning functionalities including logistic regression and model evaluation
- `functools`: For functional programming utilities

## Dataset

The dataset consists of 150 samples of iris flowers, each with four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable is the species of the iris flower, which can be one of three classes: Iris Setosa, Iris Versicolor, and Iris Virginica.

## Steps

1. **Load the Dataset**: The dataset is loaded from a CSV file named `iris.data`.

2. **Data Exploration**:
   - Basic information and statistics of the dataset are displayed.
   - The distribution of species is visualized using a bar plot.
   - Relationships among features are explored using pair plots and correlation matrices.

3. **Data Preprocessing**:
   - Categorical variables (species) are one-hot encoded to convert them into a numerical format suitable for model training.
   - The dataset is split into training and test sets.

4. **Feature Selection**:
   - Sequential Feature Selection (forward selection) is used to identify the most significant features for model training.

5. **Model Training and Evaluation**:
   - A logistic regression model is defined and trained using the selected features.
   - Predictions are made on the test set.
   - Model performance is evaluated using accuracy, precision, recall, and F1-score for each class.

