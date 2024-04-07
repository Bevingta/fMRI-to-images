# Decrypting the Brain: Extracting Visual Information from fMRI Data
---
### Description
- This study was realized as the final project for the course "CSCI 3397/PSYC 3317: Biomedical Image Analysis" at Boston College. Its scope is to reconstruct images starting from fmri data, reversing in this way the process that happens in the human brain as a response to visual stimuli. The main reference for this project was the [Brain-diffuser](https://github.com/ozcelikfu/brain-diffuser) model, that we aim to reproduce and possibly modify.
---
### Instructions
The instructions are mainly reported from the Brain-diffuser project, rewritten including what we did, the complications found, and the relative solutions that consented us to proceed.
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
- Fourth step. Download "COCO_73k_annots_curated.npy" file from [HuggingFace NSD](https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main) and place it in the 'annots' folder;
- Fifth step. Prepare NSD data for the Reconstruction Task:
        ```python
        cd data
        python prepare_nsddata.py -sub 1
        python prepare_nsddata.py -sub 2
        python prepare_nsddata.py -sub 5
        python prepare_nsddata.py -sub 7
  **Note:** this may cause problems due to the large dimension of the data. If you obtain an error related to the impossibility of store large arrays in the RAM, you can open the `prepare_nsddata.py` file and change the code line
        ```python
        stim = f_stim['imgBrick']
to
        ```python
        stim = f_stim['imgBrick'][:,::2,::2]
- Sixth step.
  
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
  Then, in order to extract the features in the way suggested by the authors and upload the corresponding files in the shared Google Drive, I found it was necessary to install pytorch with CUDA support.
- I wrote the "Description" and "References" sections of the ReadMe;

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
