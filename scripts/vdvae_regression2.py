import sys
import numpy as np
import sklearn.linear_model as skl
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

import tensorflow as tf #ADDED BY DREW
from tensorflow.keras import Sequential #ADDED BY DREW
from tensorflow.keras.layers import Dense #ADDED BY DREW
from sklearn.preprocessing import StandardScaler #ADDED BY DREW

import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]

nsd_features = np.load('data/extracted_features/subj{:02d}/nsd_vdvae_features_31l.npz'.format(sub))
train_latents = nsd_features['train_latents']
test_latents = nsd_features['test_latents']

train_path = 'data/processed_data/subj{:02d}/nsd_train_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
train_fmri = np.load(train_path)
test_path = 'data/processed_data/subj{:02d}/nsd_test_fmriavg_nsdgeneral_sub{}.npy'.format(sub,sub)
test_fmri = np.load(test_path)

## Preprocessing fMRI

train_fmri = train_fmri/300
test_fmri = test_fmri/300


norm_mean_train = np.mean(train_fmri, axis=0)
norm_scale_train = np.std(train_fmri, axis=0, ddof=1)
train_fmri = (train_fmri - norm_mean_train) / norm_scale_train
test_fmri = (test_fmri - norm_mean_train) / norm_scale_train

print(np.mean(train_fmri),np.std(train_fmri))
print(np.mean(test_fmri),np.std(test_fmri))

print(np.max(train_fmri),np.min(train_fmri))
print(np.max(test_fmri),np.min(test_fmri))

num_voxels, num_train, num_test = train_fmri.shape[1], len(train_fmri), len(test_fmri)

###################################################################################################
#ADDED BY DREW

# Standardize features
scaler = StandardScaler()
train_fmri_scaled = scaler.fit_transform(train_fmri)
test_fmri_scaled = scaler.transform(test_fmri)

# Define the model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(train_fmri_scaled.shape[1],)),
    Dense(train_latents.shape[1])  # Output layer with the same number of units as the number of latent features
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model in mini-batches
batch_size = 1000
epochs = 10  # Number of epochs per mini-batch
print("Training the model...")
for i in range(0, len(train_fmri_scaled), batch_size):
    X_batch = train_fmri_scaled[i:i+batch_size]
    y_batch = train_latents[i:i+batch_size]
    
    # Fit the model with the mini-batch
    model.fit(X_batch, y_batch, epochs=epochs, verbose=0)
    print(f"Processed {i + len(X_batch)} samples out of {len(train_fmri_scaled)}", end="\r")

print("Training completed.")

# Evaluate the model
print("Evaluating model...")
score = model.evaluate(test_fmri_scaled, test_latents)
print("MSE Score:", score)

# Predict latent variables
print("Predicting latent variables...")
pred_test_latent = model.predict(test_fmri_scaled)

# Standardize predicted latent variables
print("Standardizing predicted latent variables...")
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)

# Scale predicted latent variables back to original scale
print("Scaling predicted latent variables back to original scale...")
pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)

# Save predicted latent variables
print("Saving predicted latent variables...")
np.save('data/predicted_features_drew_version/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub, sub), pred_latents)
print("Saved")

# Extract weights and biases from the model
weights = []
biases = []
for layer in model.layers:
    weights.append(layer.get_weights()[0])
    biases.append(layer.get_weights()[1])

# Create a dictionary to save weights and biases
datadict = {
    'weights': weights,
    'biases': biases
}

with open('data/regression_weights_drew_version/subj{:02d}/vdvae_regression_weights.pkl'.format(sub),"wb") as f:
  pickle.dump(datadict,f)

###################################################################################################
