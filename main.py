import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import os #added in order to allow us to open our local image folder
import re # regex my beloved -  READ ME - let me know if you want me to explain whats going on with this library

labels = [] # a list that will contain all of the labels of our inputs by index. i might run into trouble when I have to put this into the dataloader but I will think of a solution when i get there

images = os.listdir("Fingers")[:100]  #READ ME  os.listdir pulls the first 100 file names from the local folder "Fingers". if your version of the data set is not both called "Fingers" and within the CMPM17-FINAL.... folder, this wont work. Hit me up if you run into any trouble!

for idx, file in enumerate(images):

    img = Image.open("Fingers\\" + file) # file only stores the file name, so in order to open the image we have to replicate the relative path. all my images are in the "Fingers" folder, this wont work if your folder is not named Fingers or you are on mac (:skull:)
    
    if re.search("_0R", file): #possibly the most INEFFECIENT solution possible but i might think of something better if i wish upon a shooting star
        labels.append("RH - 0")
    elif re.search("_1R", file):
        labels.append("RH - 1 ")
    elif re.search("_2R", file):
        labels.append("RH - 2")
    elif re.search("_3R", file):
        labels.append("RH - 3")
    elif re.search("_4R", file):
        labels.append("RH - 4")
    elif re.search("_5R", file):
        labels.append("RH - 5")
    elif re.search("_0L", file):
        labels.append("LH - 0")
    elif re.search("_1L", file):
        labels.append("LH - 1")
    elif re.search("_2L", file):
        labels.append("LH - 2")
    elif re.search("_3L", file):
        labels.append("LH - 3")
    elif re.search("_4L", file):
        labels.append("LH - 4")
    elif re.search("_5L", file):
        labels.append("LH - 5")
    else:
        labels.append("third more sinister option")
    pics = plt.subplot(5,20,idx+1)
    pics.imshow(img)
    pics.set_title(labels[idx])
    pics.axis('off')
plt.tight_layout()
plt.show()
