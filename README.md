# Decrypting the Brain: Extracting Visual Information from fMRI Data
---
### Description
- This study was realized as the final project for the course "CSCI 3397/PSYC 3317: Biomedical Image Analysis" at Boston College. Its scope is to reconstruct images starting from fmri data, reversing in this way the process that happens in the human brain as a response to visual stimuli. The main reference for this project was the [Brain-diffuser](https://github.com/ozcelikfu/brain-diffuser) model, that we aim to reproduce and possibly modify.
---
### Instructions
The instructions are mainly reported from the Brain-diffuser project, rewritten including what we did, the complications found, and the relative solutions that consented us to proceed.

**Obtaining and processing the data**

- First step. Copy the Brain-diffuser repository, in order to have the folders appropriately ordered;
- Second step. Obtain access to NSD data. In order to do so, it is needed to:
  
  - Watch the useful tutorials [AWS CLI Tutorial](https://www.youtube.com/watch?v=Rp-A84oh4G8&t=39s) and [AWS CLI for Beginners](https://www.youtube.com/watch?v=9oYd5KQM8AQ&t=315s);
  - Create a Root User AWS account;
  - Go in the IAM console and create an IAM user with Administratoraccess permission;
  - Access from the IAM user account;
  - Create access keys;
  - Run `aws configure` on the terminal or anaconda powershell prompt;
  - Provide the access keys you just created.
  
- Third step. Download the NSD Data fromNSD AWS Server:
    ```python
    cd data
    python download_nsddata.py
    ```
  
- Fourth step. Download "COCO_73k_annots_curated.npy" file from [HuggingFace NSD](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main) and place it in the `annots` folder;
- Fifth step. Prepare NSD data for the Reconstruction Task:
    ```python
    cd data
    python prepare_nsddata.py -sub 1
    python prepare_nsddata.py -sub 2
    python prepare_nsddata.py -sub 5
    python prepare_nsddata.py -sub 7
    ```
  
**Note:** this may cause problems due to the large dimension of the data. If you obtain an error related to the impossibility of store large arrays in the RAM, you can open the `prepare_nsddata.py` file and change the code line
  ```python
  stim = f_stim['imgBrick']
  ```
to  
  ```python
  stim = f_stim['imgBrick'][:,::2,::2]
  ```

**VDVAE: Obtaining and implementing the model**
        
- Sixth step. Download pretrained VDVAE model files from the following links:  
  [imagenet64-iter-1600000-log.jsonl](https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl)   
  [imagenet64-iter-1600000-model.th](https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th)  
  [imagenet64-iter-1600000-model-ema.th](https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th)  
  [imagenet64-iter-1600000-opt.th](https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th)  
  Place the files in the `vdvae/model/` folder.

- Seventh step. Extract VDVAE latent features of stimuli images for any subject 'x' using `python scripts/vdvae_extract_features.py -sub x`. In order to make this work, the presence of an NVIDIA GPU that is CUDA-compatible is necessary. Moreover, one needs to install pytorch with CUDA support, referring to the instructions in [Pytorch Website](https://pytorch.org/get-started/locally/).
- Eighth step. Train regression models from fMRI to VDVAE latent features and save test predictions using `python scripts/vdvae_regression.py -sub x`. Once again, this may cause errors related to the large dimensions of the data. It is possible to overcome this problem replacing the following line of code  
  
  ```python
  reg.fit(reduced_train_fmri, train_latents)
  ```
  
  with this block of code
  
    ```python
    print('train_fmri', train_fmri.shape, train_fmri[0])
    print('train_latents', train_latents.shape, train_latents[0])
    best_score = 0
    best_mask = np.zeros(train_fmri.shape[1], dtype=bool)
    train_fmri = train_fmri.astype(np.float32)
    np.random.seed(42)
    for _ in range(2):
      print(_)
      features_to_remove = np.random.choice(train_fmri.shape[1], 10500, replace=False)
      mask = np.ones(train_fmri.shape[1], dtype=bool)
      mask[features_to_remove] = 0
      reduced_train_fmri = train_fmri[:,mask]
      reg.fit(reduced_train_fmri, train_latents)
      score = reg.score(reduced_train_fmri, train_latents)
      if score > best_score:
        best_score = score
        best_mask = mask
    train_fmri = train_fmri[:,best_mask]
    test_fmri = test_fmri[:,best_mask]
    reg.fit(train_fmri, train_latents)
    ```

    This code is used to iteratively select a number (in this case 10500) of features at random, remove them from the training data, and fit the Ridge regression on the so obtained reduced data, obtaining the $R^2$ score. At the end of the iterations, the group of features whose removal lead to the best score is permanently removed, and the model is fitted on the remaining features. This is useful to solve the problems with memory allocation, at the cost of a reduced accuracy.
- Ninth step. Reconstruct images from predicted test features using `python scripts/vdvae_reconstruct_images.py -sub x`.

**Versatile Diffusion: Obtaining and implementing the model**

- Download pretrained Versatile Diffusion model "vd-four-flow-v1-0-fp16-deprecated.pth", "kl-f8.pth" and "optimus-vae.pth" from [HuggingFace](https://huggingface.co/shi-labs/versatile-diffusion/tree/main/pretrained_pth) and put them in `versatile_diffusion/pretrained/` folder
- Extract CLIP-Text features of captions for any subject 'x' using `python scripts/cliptext_extract_features.py` -sub x
- Extract CLIP-Vision features of stimuli images for any subject 'x' using `python scripts/clipvision_extract_features.py` -sub x
- Train regression models from fMRI to CLIP-Text features and save test predictions using `python scripts/cliptext_regression.py` -sub x
- Train regression models from fMRI to CLIP-Vision features and save test predictions using `python scripts/clipvision_regression.py` -sub x
- Reconstruct images from predicted test features using `python scripts/versatilediffusion_reconstruct_images.py` -sub x. Depending on the number of GPU devices of your machine you may want to edit this last code script, for example changing all the `cude(1)` calls to `cuda(0)`.
  
### Contributions
**Andrea**
- I participated to all the group meetings, proposing and discussing ideas, first on which model to choose and then on how to solve the arising problems;
- I read the DREAM paper and wrote a resume of it to better understand the topic. This helped to better understand the complications that could have arose in the realization of the DREAM
  project, leading to the decision of referring to a simplified version instead. This simplified version, called "Brain-diffuser", was also suggested by the authors of DREAM.
- I learned how the AWS environment works and I created my AWS Root user account and the AWS IAM user accounts for the group, obtaining the access keys to the NSD dataset for myself and the other members;
- I downloaded the data on my computer and uploaded it on a Google Drive that I then shared with the other group members to simplify their access to the files. I then mounted the Google Drive on a Colab
  Notebook and I tried to upload the data on DropBox in order to provide the other group members with further solutions to obtain the data;
- I gradually solved the problems arising from following the instructions provided by the authors of "Brain-diffuser". First, the large amount of data made running the "Brain-diffuser"
  code unfeasible, as it was creating arrays that were too big to be stored in the RAM. This was solved by downsampling the data about images, keeping all the images but fewer pixels.
  Then, in order to extract the features in the way suggested by the authors and upload the corresponding files in the shared Google Drive, I found it was necessary to install pytorch with CUDA support. Subsequently, I found a solution to the errors arising while training the regression model. I needed to reduce the number of features in order for the code to run, and I tried to achieve a trade off between a logic choice of features and efficiency, as it can be seen from the block of code described in the eighth step above. I obtained the reconstructed images.
- I wrote the "Description", "References" and "Instructions" sections of the ReadMe;
- I made a scheme that represents in a simpliefied way the functioning of the model, the dimensions of the data, and the code scripts related to the various steps of the model;
- I obtained the first images reconstructed with Versatile Diffusion, noting that the main problems were due to the scarcity of memory remained in my computer;
- I tried to reduce the batch size from 30 to 3 and the number of VDVAE layers from 31 to 10 trying to see whether we could obtain good results with a smaller memory usage;

**Drew**
- Proposal writeup
- Database proposal
- Model proposal
- Created a script to access the files through the Google Drive API

**Camille**
- CLIP research

---
### References
- The paper from which we obtained the information on the model was ["Brain-Diffuser: Natural scene reconstruction from fMRI signals using generative latent diffusion"](https://arxiv.org/abs/2303.05334), by Furkan Ozcelik and Rufin VanRullen
- Our work was initially inspired by a related project, described in [DREAM: Visual Decoding from Reversing Human Visual System](https://arxiv.org/pdf/2310.02265.pdf), by Weihao Xia,  Raoul de Charette,  Cengiz Öztireli, and  Jing-Hao Xue1.
- Useful tutorials for how to download data from AWS are [AWS CLI Tutorial](https://www.youtube.com/watch?v=Rp-A84oh4G8&t=39s) by 
Stephane Maarek and [AWS CLI for Beginners](https://www.youtube.com/watch?v=9oYd5KQM8AQ&t=315s) by BrainTrust Digital.
- Dataset used in the studies are obtained from [Natural Scenes Dataset](https://naturalscenesdataset.org/)
