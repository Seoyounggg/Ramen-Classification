import os
import sys
import random

train_path = './train/'
validation_path = './val/'
test_path = './test/'
num_sample = 100

if not os.path.exists(validation_path):
    os.mkdir(validation_path)
else:
    print('error path already exists!!')

if not os.path.exists(test_path):
    os.mkdir(test_path)
else:
    print('error path already exists!!')

folder_list = os.listdir(train_path)
print('folder list : {}'.format(folder_list))

for dir_path in folder_list:
    if not os.path.exists(os.path.join(validation_path, dir_path)):
        os.mkdir(os.path.join(validation_path, dir_path))
    load_path = os.path.join(train_path, dir_path)
    data_list = os.listdir(load_path)
    sample_list = random.sample(data_list, num_sample)
    for filename in samplelist:
        os.renames(os.path.join(load_path, filename), os.path.join(validation_path, dir_path, filename))

print('Sampling completed')
