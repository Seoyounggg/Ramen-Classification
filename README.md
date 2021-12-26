# Basic_Classification

During the summer of 2018, I studied Deep Learning with a group of people from Kwangwoon University. We watched Deep Learning lectures from youtube by hunkim. During hunkim's lectures, the images that we used were mnist. To use the mnist, all we had to do was merely load images through importing datasets from PyTorch using datasets. So we did not have a chance to load images other than the mnist dataset.

For that reason, we decided to make a model that can distinguish four different types of ramens. In this basic_classification project, our goal was not to make a unique and outstanding model. We focused on making our dataset and using them.

Each team member made their own model, and I made a model using a pre-trained VGG16 to classify ramens implemented by PyTorch Framework.

# Experiments 

Test accuracy before training the model.  
(이미지)

Test accuracy after the transfer learning.  
(이미지)

* Training the model with only two epochs was enough as its validation accuracy reached ~~.
* The weight of the model was saved when the validation accuracy was the highest.
* The test accuracy after the transfer learning was 99%. 


# Conclusion

Our project was meaningful that we created our own image dataset and train a model with it.
1. ImageFolder
-> When you want to load your custom dataset, you can use imagefolder function under torchvision.datasets.
2. Transfer learning
-> Through transfer learning from VGG16 model, the model made a successful results with small amount of data. I only removed the last layer of the original model and added another layer to make the output into the size of four.
3. Importance of the dataset
-> Even though the number of the dataset was not huge, the model had an accuracy of 99% on our dataset. The model itself might have learned well and made feature vectors nicely, but the high accuracy might indicate that the datasets used in training and testing might have been too similar. We tried our best to collect generalized dataset, but it might have not enough.
