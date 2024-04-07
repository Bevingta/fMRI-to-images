# Decrypting the Brain: Extracting Visual Information from fMRI Data
---
### Description

---
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

**Drew**
- Proposal writeup
- Database proposal
- Model proposal
- Created a script to access the files through the Google Drive API

**Camille**
- CLIP research
