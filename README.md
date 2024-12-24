# Here's a complex Python program for educational purposes that demonstrates how to detect credit card fraud using machine learning:

This program does the following:

1. It loads the credit card fraud dataset from a CSV file using Pandas.
2. It separates the features (X) and the target variable (y) from the dataset.
3. It splits the data into training and testing sets using train_test_split() from scikit-learn.
4. It creates a Random Forest classifier with 100 estimators and trains it on the training data.
5. It makes predictions on the testing set using the trained Random Forest classifier.
6. It evaluates the model's performance using accuracy, precision, recall, and F1-score metrics.

Note: Make sure to have the creditcard.csv file in the same directory as the Python script or provide the correct file path.

This program demonstrates the basic steps for detecting credit card fraud using machine learning.
In practice, you would need to preprocess the data, handle missing values, and explore additional features to improve the model's performance. 
Additionally, you can use techniques like undersampling or oversampling to handle class imbalance in the dataset.
