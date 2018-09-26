import os
import sys
import random

trainpath = './train/'
validationpath = './val/'
num_sample = 25

if not os.path.exists(validationpath):
    os.mkdir(validationpath)
else:
    print('error path already exists!!')
    sys.exit(0)

folderlist = os.listdir(trainpath)
print('folder list : {}'.format(folderlist))
for dirpath in folderlist:
    if not os.path.exists(os.path.join(validationpath, dirpath)):
        os.mkdir(os.path.join(validationpath, dirpath))
    loadpath = os.path.join(trainpath,dirpath)
    datalist = os.listdir(loadpath)
    samplelist = random.sample(datalist, num_sample)
    for filename in samplelist:
        os.renames(os.path.join(loadpath, filename), os.path.join(validationpath, dirpath, filename))


print('Sampling completed')
