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

all_imgs = [os.path.join("Fingers\\", img) for img in images] #all of the actual images will go in here ; updated so adding images to a list is more memory efficient

#training data
training_images = all_imgs[:15750]
training_labels = data[:15750]

#testing data
testing_images = all_imgs[15750:]
testing_labels = data[15750:]

#transform sequences
finger_transforms = v2.Compose([
    v2.ToTensor(),

    #helps to generalize hand size/positioning
    v2.RandomRotation(degrees=[-180, 180]),
    v2.RandomPerspective(distortion_scale=0.6, p=0.8),
    v2.RandomApply([v2.RandomAffine(degrees=[0,0],translate=(0.3,0.3),scale=(0.5,1.5))], p=0.8),
    #helps to minimize background noise and lead the model to focus on the hand details
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5.0))], p=0.5),
    v2.RandomApply([v2.RandomCrop(size=(96,96))],p=0.5),
    v2.RandomErasing(p=0.5, scale=(0.05,0.2), ratio=(0.3,3.3), value="random"),

    v2.ToPILImage()
])


class FingerData(Dataset): 
    def __init__(self, features, labels, transform=None): #Added transform parameter. Used to pass transforms to only the testing data.
        self.features = features
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.features) #how long - the tensors should be of equal length anyway
    def __getitem__(self, index):
        img = self.features[index]
        label = self.labels[index]
        if self.transform != None: #checks if transform is passed or not
            img = self.transform(img) 
        return (img, label) #returns augmented image and the corresponding label



#WILL USE THIS DATALOADER TO TRAIN DATA
finger_train = FingerData(training_images, training_labels, transform=finger_transforms)
finger_dl_train = DataLoader(finger_train, batch_size=64, shuffle=True)


#WILL USE THIS DATALOADER TO TEST DATA
finger_test = FingerData(testing_images, testing_labels) #don't define transform, we want to keep original images when testing.
finger_dl_test = DataLoader(finger_test, batch_size=64, shuffle=True)

print("no errors! yippie")