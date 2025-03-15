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
import wandb
import torch.nn.functional as F

images = os.listdir("Fingers")
labels = [] # a list that will contain all of the labels of our inputs by index. i might run into trouble when I have to put this into the dataloader but I will think of a solution when i get there
run = wandb.init(project="Hand Detector CMPM17 Final", name="model_with_accuracy2")

# Check if CUDA (GPU) is available; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Debugging to confirm GPU usage


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

all_imgs = [os.path.join("Fingers\\", img) for img in images] #all of the actual images will go in here ; updated so adding images to a list is more memory efficient

#training data
training_images = all_imgs[:15750]
training_labels = data[:15750]

#testing data
testing_images = all_imgs[15750:18375]
testing_labels = data[15750:18375]

validation_images = all_imgs[18375:]
validation_labels = data[18375:]

#transform sequences
finger_transforms = v2.Compose([
    v2.ToTensor(),
    #helps to generalize hand size/positioning
    v2.RandomRotation(degrees=[-180, 180]),
    v2.RandomPerspective(distortion_scale=0.5, p=0.8),
    v2.RandomApply([v2.RandomAffine(degrees=[0,0],translate=(0.1,0.1),scale=(0.9,1.5))], p=0.8),
    #helps to minimize background noise and lead the model to focus on the hand details
    v2.RandomApply([v2.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 5.0))], p=0.5),
    v2.RandomApply([v2.RandomCrop(size=(96,96))],p=0.3),
    #generalizes brightness
    v2.RandomApply([v2.ColorJitter(brightness=(0.1,1.5), contrast=(0.5))],p=0.8),
    v2.Resize((128,128))
])
test_transform = v2.Compose([  #things get kind of weird here in order to not apply transformations to the testing images but trust the process
    v2.ToTensor()
])

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        :param alpha: Weighting factor for class imbalance (set to 1 if not needed)
        :param gamma: Focusing parameter (higher = more focus on hard examples)
        :param reduction: 'mean' (default) or 'sum' for loss aggregation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: Predictions (logits before softmax for multi-class, probability for binary)
        :param targets: Ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # Compute standard CE loss
        p_t = torch.exp(-ce_loss)  # Get softmax probabilities
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss  # Apply focal weighting

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
class FingerData(Dataset): 
    def __init__(self, features, labels, transform=None): #Added transform parameter. Used to pass transforms to only the training data.
        self.features = features
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.features) #how long - the tensors should be of equal length anyway
    def __getitem__(self, index):
        img = Image.open(self.features[index])
        label = self.labels[index]
        if self.transform != None: #checks if transform is passed or not
            img = self.transform(img)
        else:
            img = test_transform(img) #testing images still needs to be a tensor
        return (img, label) #returns augmented image and the corresponding label
    
filters = 32

class MyModel(nn.Module): #our ml model class, inherits from some class idk the specifics of :3

    def __init__(self):
        super().__init__()
        self.activation = nn.Sigmoid() 
        self.activation2 = nn.ReLU() 
        self.softmax = nn.Softmax()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(32,64, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(64,64, kernel_size=3, padding=1) #2 layers are NOT enough for what we are trying to do
        self.layer4 = nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.layer9 = nn.Linear(16384,2048)
        self.layer10 = nn.Linear(2048,1024)
        self.layer11 = nn.Linear(1024,512)
        self.layer12 = nn.Linear(512,12)
    
    def forward(self, input):
        partial = self.layer1(input)
        partial = self.activation2(partial) 
        partial = self.layer2(partial)
        partial = self.activation2(partial)
        partial = self.maxpool(partial)
        partial = self.layer3(partial)
        partial = self.activation2(partial)
        partial = self.maxpool(partial)
        partial = self.layer4(partial)
        partial = self.activation2(partial)
        partial = self.maxpool(partial)
        partial  = torch.flatten(partial, start_dim=1)
        partial = self.layer9(partial)
        partial = self.activation2(partial)
        partial = self.layer10(partial)
        partial = self.activation2(partial)
        partial = self.layer11(partial)
        partial = self.activation2(partial)
        output = self.layer12(partial)
        #output = self.softmax(output)
        return output # returns output 
    
#WILL USE THIS DATALOADER TO TRAIN DATA
finger_train = FingerData(training_images, training_labels, transform=finger_transforms)
finger_dl_train = DataLoader(finger_train, batch_size=64, shuffle=True)

#WILL USE THIS DATALOADER TO TEST DATA
finger_test = FingerData(testing_images, testing_labels) #don't define transform, we want to keep original images when testing.
finger_dl_test = DataLoader(finger_test, batch_size=64, shuffle=True)

#VALIDATION
finger_val = FingerData(validation_images, validation_labels)
finger_dl_val = DataLoader(finger_val, batch_size=64, shuffle=True)

#le model

model = MyModel().to(device)

#loss fn/optimizer initalization

lossfn = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)

#Used to observe augmented/test images

'''
to_pil = v2.ToPILImage()
for batch in finger_dl_train:
    images, labels = batch
    # Display each image in the batch
    for img_tensor in images:
        pil_img = to_pil(img_tensor)
        plt.imshow(pil_img, "grey")
        plt.axis("off")
        plt.show()
'''


print("training in progress...")
for epoch in range(20):
    model.train()
    train_loss = 0.0
    train_correct = 0.0
    train_total = 0.0
    print("starting new batching")
    for batch in finger_dl_train: # training data loop
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        labels = labels.argmax(dim=1) #converts one hot encoded back into classification (0-11)
        pred = model(images)
        #print("IMAGE TENSOR: " + str(pred.shape) + ", LABEL TENSOR: " + str(labels.shape)) #we could print the actual values for each by just dropping the .shape at the end of each image and label, but this is nicer in the terminal for now
        loss = lossfn(pred, labels)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        #calculating accuracy
        pred_fingers = torch.argmax(pred, dim=1)
        train_correct += (pred_fingers == labels).sum().item()
        train_total += labels.size(0)

    #end batch, calculate testing loss + accuracy
    avg_train_loss = train_loss/len(finger_dl_train)
    train_accuracy = train_correct/train_total*100

    #validation loop
    val_loss = 0.0
    val_correct = 0.0
    val_total = 0.0 
    with torch.no_grad():
        for images, labels in finger_dl_val:
            images, labels = images.to(device), labels.to(device)
            labels=labels.argmax(dim=1) #converts one hot encoded back into classification (0-11)
            val_pred=model(images)
            loss = lossfn(val_pred, labels)
            val_loss += loss.item()
            #calculating accuracy
            pred_fingers_val = torch.argmax(val_pred, dim=1)
            val_correct += (pred_fingers_val == labels).sum().item()
            val_total += labels.size(0)

    #end batch, calculate validation loss + accuracy
    avg_val_loss = val_loss/len(finger_dl_val)
    val_accuracy = val_correct/val_total*100
    
    #adds a datapoint of training and validation loss, along with epoch and accuracies
    print(f"Epoch {epoch+1}, Train Accuracy: {train_accuracy:.4f}% , Validation Accuracy: {val_accuracy:.4f}%")
    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "train_acc":train_accuracy , "val_acc":val_accuracy})

print("final loss for training model:", loss)

#testing data loop
for epoch in range(20):
    model.eval()
    test_loss = 0.0
    test_correct = 0.0
    test_total = 0
    print("starting new batching for testing. epoch: ", epoch+1)
    for images, labels in finger_dl_test:
        images, labels = images.to(device), labels.to(device)
        labels = labels.argmax(dim=1) #converts one hot encoded back into classification (0-11)
        pred = model(images)
        loss = lossfn(pred, labels)
        
        test_loss += loss.item()
        #compute accuracy
        pred_fingers_test = torch.argmax(pred, dim=1)
        test_correct += (pred_fingers_test == labels).sum().item()
        test_total += labels.size(0)

    avg_test_loss = test_loss/len(finger_dl_test)
    test_accuracy = test_correct/test_total*100

    #records test loss + accuracy
    print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy:.4f}%")
    wandb.log({"test loss": avg_test_loss, "test_acc": test_accuracy})

print("model finished running! congrats on waiting this long")