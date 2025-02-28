import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
import os #added in order to allow us to open our local image folder
import re # regex my beloved -  READ ME - let me know if you want me to explain whats going on with this library

images = os.listdir("Fingers")
labels = [] # a list that will contain all of the labels of our inputs by index. i might run into trouble when I have to put this into the dataloader but I will think of a solution when i get there

for image in images:
    if re.search("_0R", image): #possibly the most INEFFECIENT solution possible but i might think of something better if i wish upon a shooting star
        labels.append("RH - 0")
    elif re.search("_1R", image):
        labels.append("RH - 1 ")
    elif re.search("_2R", image):
        labels.append("RH - 2")
    elif re.search("_3R", image):
        labels.append("RH - 3")
    elif re.search("_4R", image):
        labels.append("RH - 4")
    elif re.search("_5R", image):
        labels.append("RH - 5")
    elif re.search("_0L", image):
        labels.append("LH - 0")
    elif re.search("_1L", image):
        labels.append("LH - 1")
    elif re.search("_2L", image):
        labels.append("LH - 2")
    elif re.search("_3L", image):
        labels.append("LH - 3")
    elif re.search("_4L", image):
        labels.append("LH - 4")
    elif re.search("_5L", image):
        labels.append("LH - 5")
    else:
        labels.append("third more sinister option") #this shouldnt be returned. if you see it something is wrong :(

values = {"vals":labels} #this is just a formatting thing so i can one hot encode the labels
df = pd.DataFrame(values, list(range(0,len(labels)))) # making this into a dataframe. i am so goated
df = pd.get_dummies(df, columns=["vals"]) #one hot encoding the column
dfa = df.to_numpy(dtype='float64') #i did this to make all values floats before putting it into a tensor
data = torch.tensor(dfa, dtype=torch.float) #become a tensor now
#initalize training data

all_imgs = [] #all of the actual images will go in here
for img in images: 
    all_imgs.append(Image.open("Fingers\\" + img)) # opening each individual image and appending into this list. i dont know if it will wokr but we will see

training_images = all_imgs[:15750]
training_labels = data[:15750]

#testing data
testing_images = all_imgs[15750:]
testing_labels = data[15750:]


class FingerData(Dataset): #fortnite is a class that inherits from Dataset. It's purpose is to help me with batching my data :3 I named it fortnite because its awesome.
    def __init__(self, features, labels): #i could make it throw an error if input is not of type tensor but im the laziest mf ever so if it works it works
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features) #how long - the tensors should be of equal length anyway
    def __getitem__(self, index):
        #TRANSFORMS GO HERE
        transforms = v2.compose([])
        return(self.features[index], self.labels[index]) #value where


