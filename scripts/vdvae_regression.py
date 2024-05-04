import sys
import numpy as np
import sklearn.linear_model as skl
import argparse
import pickle # ADDED BY ANDREA
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
print('TEST', test_fmri.shape, test_latents.shape) #ADDED BY ANDREA

## latents Features Regression
print('Training latents Feature Regression')

reg = skl.Ridge(alpha=8000, max_iter=5000, fit_intercept=True)

#START OF BLOCK ADDED BY ANDREA
print('train_fmri', train_fmri.shape, train_fmri[0]) #ADDED BY ANDREA
print('train_latents', train_latents.shape, train_latents[0]) # ADDED BY ANDREA
#SOLUTION 1
best_score = 0 #ADDED BY ANDREA
best_mask = np.zeros(train_fmri.shape[1], dtype=bool) #ADDED BY ANDREA
#FOR LOOP BLOCK ADDED BY ANDREA
train_fmri = train_fmri.astype(np.float32)
np.random.seed(42)
for _ in range(3):
  print(_)
  features_to_remove = np.random.choice(train_fmri.shape[1], 10500, replace=False)
  mask = np.ones(train_fmri.shape[1], dtype=bool)
  mask[features_to_remove] = 0
  reduced_train_fmri = train_fmri[:,mask]
  reg.fit(reduced_train_fmri, train_latents) #MODIFIED BY ANDREA. THERE WAS train_fmri INSTEAD OF reduced_train_fmri AND IT WAS NOT IN A FOR LOOP
  score = reg.score(reduced_train_fmri, train_latents)
  if score > best_score:
    best_score = score
    best_mask = mask
train_fmri = train_fmri[:,best_mask]#ADDED BY ANDREA
test_fmri = test_fmri[:,best_mask]#ADDED BY ANDREA
reg.fit(train_fmri, train_latents)#ADDED BY ANDREA
#SOLUTION 3
# import torch
# import torch.nn as nn
# import torch.optim as optim

# # Define the model
# class RidgeRegression(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(RidgeRegression, self).__init__()
#         self.linear = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         return self.linear(x)

# model = RidgeRegression(input_size=train_fmri.shape[1], output_size=train_latents.shape[1])

# # Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)  # L2 penalty as weight_decay
# # Convert inputs and targets to tensors
# inputs = torch.tensor(train_fmri, dtype=torch.float32)
# targets = torch.tensor(train_latents, dtype=torch.float32)
# # Training loop
# for epoch in range(10):
#     print(epoch)

#     # Forward pass
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)

#     # Backward and optimize
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#SOLUTION 2
# Define cross-validation strategy
# from sklearn.model_selection import KFold
# cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

# # Initialize RFECV with the Ridge regression model and the cross-validation strategy
# from sklearn.feature_selection import RFECV
# rfecv = RFECV(estimator=reg, step=1, cv=cv_strategy, scoring='r2', min_features_to_select=1)

# # Fit RFECV to your data
# rfecv.fit(train_fmri, train_latents)

# # After fitting, rfecv.support_ shows the features selected
# selected_features_mask = rfecv.support_
# test_fmri = test_fmri[:,selected_features_mask]

# Define batch size
# batch_size = 1000  # Adjust based on your memory constraints

# # Incremental fitting
# for i in range(0, reduced_train_fmri.shape[0], batch_size):
#     batch_fmri = reduced_train_fmri[i:i+batch_size]
#     batch_latents = train_latents[i:i+batch_size]
#     reg.partial_fit(batch_fmri, batch_latents)

# # Score the model
# score = reg.score(reduced_train_fmri, train_latents)
#END OF BLOCK ADDED BY ANDREA
pred_test_latent = reg.predict(test_fmri)
std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent,axis=0)) / np.std(pred_test_latent,axis=0)
pred_latents = std_norm_test_latent * np.std(train_latents,axis=0) + np.mean(train_latents,axis=0)
print(reg.score(test_fmri,test_latents))
np.save('data/predicted_features/subj{:02d}/nsd_vdvae_nsdgeneral_pred_sub{}_31l_alpha50k.npy'.format(sub,sub),pred_latents)


datadict = {
    'weight' : reg.coef_,
    'bias' : reg.intercept_,

}

with open('data/regression_weights/subj{:02d}/vdvae_regression_weights.pkl'.format(sub),"wb") as f:
  pickle.dump(datadict,f)