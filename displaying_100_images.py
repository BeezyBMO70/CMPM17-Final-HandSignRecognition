'''
images100 = os.listdir("Fingers")[:100]  #READ ME  os.listdir pulls the first 100 file names from the local folder "Fingers". if your version of the data set is not both called "Fingers" and within the CMPM17-FINAL.... folder, this wont work. Hit me up if you run into any trouble!
values = [0,0,0,0,0,0,0,0,0,0,0,0,0] # - to keep track of class distribution, in order matching order below.
for idx, file in enumerate(images100):
    img = Image.open("Fingers\\" + file) # file only stores the file name, so in order to open the image we have to replicate the relative path. all my images are in the "Fingers" folder, this wont work if your folder is not named Fingers or you are on mac (:skull:)
    if re.search("_0R", file): #possibly the most INEFFECIENT solution possible but i might think of something better if i wish upon a shooting star
        labels.append("RH - 0")
        values[0] += 1
    elif re.search("_1R", file):
        labels.append("RH - 1 ")
        values[1] += 1
    elif re.search("_2R", file):
        labels.append("RH - 2")
        values[2] += 1
    elif re.search("_3R", file):
        labels.append("RH - 3")
        values[3] += 1
    elif re.search("_4R", file):
        labels.append("RH - 4")
        values[4] += 1
    elif re.search("_5R", file):
        labels.append("RH - 5")
        values[5] += 1
    elif re.search("_0L", file):
        labels.append("LH - 0")
        values[6] += 1
    elif re.search("_1L", file):
        labels.append("LH - 1")
        values[7] += 1
    elif re.search("_2L", file):
        labels.append("LH - 2")
        values[8] += 1
    elif re.search("_3L", file):
        labels.append("LH - 3")
        values[9] += 1
    elif re.search("_4L", file):
        labels.append("LH - 4")
        values[10] += 1
    elif re.search("_5L", file):
        labels.append("LH - 5")
        values[11] += 1
    else:
        labels.append("third more sinister option")
        values[12] += 1
    pics = plt.subplot(5,20,idx+1)
    pics.imshow(img, "gray")
    pics.set_title(labels[idx])
    pics.axis('off')
plt.tight_layout()
plt.show()
print(values)
'''