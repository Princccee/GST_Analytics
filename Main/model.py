import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import numpy as np


# Function to load the training and testing datasets
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)  # Load training data from CSV file
    test_data = pd.read_csv(test_path)  # Load testing data from CSV file
    print(f"Training data shape: {train_data.shape}")  # Print shape of training data
    print(f"Testing data shape: {test_data.shape}")  # Print shape of testing data
    return train_data, test_data  # Return the loaded datasets


# Function to plot the correlation matrix heatmap for data exploration
def plot_correlation_heatmap(data, title="Correlation Matrix Heatmap", size=(14, 8)):
    corr = data.corr()  # Calculate correlation matrix
    plt.figure(figsize=size)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)  # Plot heatmap
    plt.title(title, fontsize=16)  # Set title for the plot
    plt.show()  # Display the heatmap


# Function to remove features with low correlation to the target variable
def remove_low_correlation_features(df, target_col, threshold=0.01):
    correlation_matrix = df.corr()  # Compute correlation matrix
    target_correlation = correlation_matrix[target_col].abs()  # Get absolute correlation with the target
    features_to_keep = target_correlation[target_correlation > threshold].index.tolist()  # Keep features with correlation above threshold
    reduced_df = df[features_to_keep]  # Create a reduced dataframe with the selected features
    print(f"Reduced data shape: {reduced_df.shape}")  # Print the new shape after feature removal
    return reduced_df, features_to_keep  # Return reduced dataframe and selected features


# Function to apply SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance
def apply_smote(X, y):
    smote = SMOTE(random_state=42)  # Initialize SMOTE with a fixed random seed for reproducibility
    X_smote, y_smote = smote.fit_resample(X, y)  # Resample the dataset to balance the classes
    class_counts = Counter(y_smote)  # Count the occurrences of each class after SMOTE
    print(f"Class distribution after SMOTE: {class_counts}")  # Print the new class distribution
    return X_smote, y_smote  # Return the resampled dataset


# Function to scale the dataset using StandardScaler (mean=0, variance=1 normalization)
def scale_data(X_train, X_test):
    scaler = StandardScaler()  # Initialize the scaler
    X_train_scaled = scaler.fit_transform(X_train)  # Fit the scaler on training data and scale it
    X_test_scaled = scaler.transform(X_test)  # Use the fitted scaler to scale test data
    return X_train_scaled, X_test_scaled, scaler  # Return scaled data and scaler


# Function to build the neural network model using Keras Sequential API
def build_model(input_shape):
    model = Sequential()  # Initialize a sequential model
    model.add(Input(shape=(input_shape,)))  # Input layer with the shape of features
    model.add(Dense(128, activation='relu'))  # First dense layer with 128 neurons and ReLU activation
    model.add(Dropout(0.3))  # Dropout layer for regularization to prevent overfitting (30% drop rate)
    model.add(Dense(64, activation='relu'))  # Second dense layer with 64 neurons
    model.add(Dropout(0.3))  # Dropout layer
    model.add(Dense(32, activation='relu'))  # Third dense layer with 32 neurons
    model.add(Dropout(0.2))  # Dropout layer with 20% drop rate
    model.add(Dense(16, activation='relu'))  # Fourth dense layer with 16 neurons
    model.add(Dropout(0.2))  # Dropout layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification (sigmoid activation)
    
    # Compile the model using Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model  # Return the compiled model


# Function to train the neural network with early stopping for improved generalization
def train_model(model, X_train, y_train, batch_size=64, epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Early stopping to avoid overfitting
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                        class_weight={0: 1.0, 1: 2.5}, callbacks=[early_stopping])  # Train the model with class weighting and early stopping
    return history  # Return the training history for analysis


# Function to evaluate the model's performance on the test set
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)  # Evaluate the model's performance
    print(f"Test Loss: {test_loss}")  # Print the test loss
    print(f"Test Accuracy: {test_accuracy}")  # Print the test accuracy
    return test_loss, test_accuracy  # Return test loss and accuracy


# Function to plot the ROC (Receiver Operating Characteristic) curve
def plot_roc_curve(y_test, y_pred_proba):
    roc_auc = roc_auc_score(y_test, y_pred_proba)  # Calculate the AUC-ROC score
    print(f"AUC-ROC: {roc_auc}")  # Print the AUC-ROC score
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)  # Get false positive rate (FPR) and true positive rate (TPR) values for plotting
    plt.plot(fpr, tpr, label=f'AUC-ROC = {roc_auc:.4f}')  # Plot the ROC curve
    plt.plot([0, 1], [0, 1], linestyle='--')  # Plot the diagonal line (random chance)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()  # Display the ROC curve


# Function to print confusion matrix and classification report
def print_classification_report(y_test, y_pred_class):
    conf_matrix = confusion_matrix(y_test, y_pred_class)  # Generate the confusion matrix
    print(f"Confusion Matrix:\n{conf_matrix}")  # Print the confusion matrix
    report = classification_report(y_test, y_pred_class)  # Generate the classification report
    print(f"Classification Report:\n{report}")  # Print the classification report


# Function to make a prediction for a single row/sample
def make_single_prediction(model, scaler, row):
    row_scaled = scaler.transform(row.values.reshape(1, -1))  # Scale the input row using the fitted scaler
    prediction = model.predict(row_scaled)  # Make prediction using the model
    if prediction >= 0.5:
        print("Predicted Class: 1")  # If the prediction probability is >= 0.5, classify as class 1
    else:
        print("Predicted Class: 0")  # Otherwise, classify as class 0


# Main workflow to execute the entire ML pipeline
def main():
    # Load training and testing datasets
    train_data, test_data = load_data('Dataset/Train_60/Train_60/Training.csv', 'Dataset/Test_20/Test_20/Testing.csv')

    # Plot correlation heatmap of the training data
    plot_correlation_heatmap(train_data)

    # Remove features with low correlation to the target variable
    train_data, features_to_keep = remove_low_correlation_features(train_data, target_col='target')
    test_data = test_data[features_to_keep]  # Apply same feature selection to test data

    # Split data into features (X) and target (y)
    X_train, y_train = train_data.drop('target', axis=1), train_data['target']
    X_test, y_test = test_data.drop('target', axis=1), test_data['target']

    # Apply SMOTE to handle class imbalance
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Scale training and testing data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train_smote, X_test)

    # Build the neural network model
    model = build_model(X_train_scaled.shape[1])

    # Train the model
    train_model(model, X_train_scaled, y_train_smote)

    # Evaluate the model on test data
    evaluate_model(model, X_test_scaled, y_test)

    # Predict probabilities on the test set and plot ROC curve
    y_pred_proba = model.predict