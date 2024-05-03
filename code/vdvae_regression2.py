import sys
import numpy as np
import sklearn.linear_model as skl
from sklearn.linear_model import SGDRegressor
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

## latents Features Regression
print('Training latents Feature Regression')

#reg = skl.Ridge(alpha=50000, max_iter=10000, fit_intercept=True)
print("SGD start")
reg = SGDRegressor(alpha=50000, max_iter=10000, fit_intercept=True) #EDITED BY DREW
print("SGD end")
print("reg.fit start ")
reg.fit(train_fmri, train_latents)
print("reg.fit end ")
pred_test_latent = reg.predict(test_fmri)
print("reg.predict end")
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
print("normalization end")
pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
print("predictions end")
print(reg.score(test_fmri,test_latents))

# Iterate over batches of data
# batch_size = 1000
# print("batch set")
# for i in range(0, len(train_fmri), batch_size):
#     batch_fmri = train_fmri[i:i+batch_size]
#     batch_latents = train_latents[i:i+batch_size]
    
#     # Reshape batch_latents to 1D array
#     batch_latents_1d = batch_latents.flatten()
    
#     # Fit the model on the current batch
#     reg.partial_fit(batch_fmri, batch_latents_1d)

#     if i%5 == 0:
#        print(i)

# # Predict on test data in batches
# batch_predictions = []
# for i in range(0, len(test_fmri), batch_size):
#     batch_test_fmri = test_fmri[i:i+batch_size]
#     batch_pred_test_latent = reg.predict(batch_test_fmri)
#     batch_predictions.append(batch_pred_test_latent)

#     if i%5 == 0:
#       print(i)

# # Concatenate batch predictions
# pred_test_latent = np.concatenate(batch_predictions)
# print("concatenated")

# # Calculate standardization
# std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / np.std(pred_test_latent, axis=0)
# print("normalized")

# # Calculate final predictions
# pred_latents = std_norm_test_latent * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)
# print("final calcualted")

# # Evaluate model
# score = reg.score(test_fmri, test_latents)
# print(score)

np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)


datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,

}

with open('data/regression_weights/subj{:02d}/vdvae_regression_weights.pkl'.format(sub),"wb") as f:
  pickle.dump(datadict,f)