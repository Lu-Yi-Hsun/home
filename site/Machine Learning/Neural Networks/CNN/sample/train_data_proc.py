#python


from PIL import Image
from os import listdir
import numpy as np
import os
def image(path):
    im = Image.open(path) #Can be many different formats.
    pix = im.load()
    #print (im.size) #Get the width and hight of the image for iterating over
    index=0
    strr=""
    for i in range(2,30):
        for j in range(2,30):
            strr=strr+str(pix[i,j]/255)+"\t"
             
    return strr


#"DevanagariHandwrittenCharacterDataset/Train/character_1_ka/1340.png"
path = listdir("DevanagariHandwrittenCharacterDataset/Train")
print(path)
train_image=""
index=0

train_class=""
classindex=0
for pa in path:
    fil = listdir("DevanagariHandwrittenCharacterDataset/Train/"+pa)
    tem=""
    for g in range(46):
        if classindex==g:
            tem=tem+"1"+"\t"
        else :
            tem=tem+"0"+"\t"


    for fi in fil:
        train_image=train_image+image("DevanagariHandwrittenCharacterDataset/Train/"+pa+"/"+fi)+"\n"
        index=index+1
        
        train_class=train_class+tem+"\n"
        #print(classfile)
        #os.system("clear")
        #print(index)
    classindex=classindex+1

f_data=open('train_data.ml','w')
f_label=open('train_label.ml','w')
###########################       
f_data.write(train_image)

f_label.write(train_class)
#print (np.array(ans))
          


 