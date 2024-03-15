import mne 
import warnings
import numpy as np

from google.colab import drive

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  select_time_window, transform_for_classificator

np.random.seed(23)

mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning ) 