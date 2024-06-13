import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Collection
def load_nsl_kdd_data():
    column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
                    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
                    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
                    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"]

    train_data = pd.read_csv("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt", header=None, names=column_names)
    test_data = pd.read_csv("https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt", header=None, names=column_names)
    
    return train_data, test_data

# Step 2: Data Preprocessing
def preprocess_data(data):
    # Drop the difficulty column as it's not needed
    data = data.drop(['difficulty'], axis=1)
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    data = pd.get_dummies(data, columns=categorical_cols)
    
    # Encode the attack labels
    label_encoder = LabelEncoder()
    data['attack'] = label_encoder.fit_transform(data['attack'])
    
    # Separate features and labels
    X = data.drop('attack', axis=1)
    y = data['attack']
    
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoder


# Example data (replace with your actual data)
epochs = range(1, 6)  # Replace with the number of epochs you have
train_loss = [0.2304, 0.1931, 0.1875, 0.1836, 0.1805]
val_loss = [0.1965, 0.1908, 0.1866, 0.1838, 0.1819]
train_acc = [0.9295, 0.9376, 0.9387, 0.9391, 0.9395]
val_acc = [0.9377, 0.9387, 0.9393, 0.9398, 0.9401]

# Plotting loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, 'bo-', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Show plot
plt.tight_layout()
plt.show()


# Step 3: Feature Extraction
def extract_features(X):
    pca = PCA(n_components=30)  # Adjust the number of components as needed
    X_pca = pca.fit_transform(X)
    return X_pca, pca

# Step 4: Model Selection & Training
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 2, activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1, auc_roc

# Step 5: Continuous Learning
def continuous_learning(model, new_data, scaler, pca):
    # Preprocess new data
    new_data_scaled = scaler.transform(new_data)
    new_data_pca = pca.transform(new_data_scaled)
    
    # Split new data into features and labels
    X_new = new_data_pca[:, :-1]
    y_new = new_data_pca[:, -1]
    X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
    
    # Retrain the model with new data
    model.fit(X_new, y_new, epochs=5)

# Main function to run the entire process
def main():
    # Load dataset
    train_data, test_data = load_nsl_kdd_data()

    # Preprocess the data
    X_train_scaled, y_train, scaler, label_encoder = preprocess_data(train_data)
    X_test_scaled, y_test, _, _ = preprocess_data(test_data)

    # Extract features using PCA
    X_train_pca, pca = extract_features(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Reshape data for Conv1D input
    X_train_pca = X_train_pca.reshape((X_train_pca.shape[0], X_train_pca.shape[1], 1))
    X_test_pca = X_test_pca.reshape((X_test_pca.shape[0], X_test_pca.shape[1], 1))

    # Build and train the model
    input_shape = (X_train_pca.shape[1], 1)
    model = build_model(input_shape)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # Train the model
    model.fit(X_train_pca, y_train, epochs=20, validation_data=(X_test_pca, y_test), callbacks=[early_stopping])

    # Evaluate the model
    accuracy, precision, recall, f1, auc_roc = evaluate_model(model, X_test_pca, y_test)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC-ROC: {auc_roc}')

    # Example of continuous learning with new data
    continuous_learning(model, X_test_scaled, scaler, pca)

# Run the main process
if __name__ == "__main__":
    main()
