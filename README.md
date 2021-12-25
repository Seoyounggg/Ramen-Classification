# Basic_Classification

During the summer of 2018, I studied Deep Learning with a group of people from Kwangwoon University. We watched Deep Learning lectures from youtube by hunkim. During hunkim's lectures, the images that we used were mnist. To use the mnist, all we had to do was merely load images through importing from datasets in PyTorch. So we did not have a chance to load images other than the mnist dataset.

For that reason, we decided to make a model that can distinguish four different types of ramens. In this basic_classification project, our goal was not to make a unique and outstanding model. We focused on making our dataset and using them.

Each team member made their own model, and I made a model using a pre-trained VGG16 to classify ramens implemented by PyTorch Framework.

# Experiments 

train 데이터로 학습하기 전 성공률(vgg16 모델로 classification한 결과)
(이미지)

데이터로 학습한 후 
(이미지)

* epoch 2개만으로도 테스트 성공률이 높음
* --> 300개의 데이터만으로도 충분한 학습 or train 데이터와 test 데이터가 너무 유사함


# Conclusion

Our project was meaningful that we created our own image dataset and train a model with it.
1. ImageFolder
2. transfer learning
3. 데이터의 수와 다양성
