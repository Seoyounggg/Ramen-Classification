# Basic_Classification

During summer in 2018, I had study with people from Kwangwoon University. We watched Deep Learning lectures from youtube by hunkim. During hunkim's lectures, images that we used were mnist. To use the mnist, all we had to do was loading. So we didn't have chance to learn how to load images that we took with our phones.

For that reason, we decided to make a model that distinguish 4 different types of ramens. In this basic_classification project, our goal was not on making unique and outstanding model. We focused on making our own training datas and using them.

My team consist of four people, and each of them made their own model.

I used pretrained VGG16 to classify ramens.

# Experiments 

(성공률 이미지 넣기)

그 결과로 성공률은 98% 였다.

Furthermore, we checked if it's possible to classify ramen's labels in real time.

The experiment showed that if we put the items under same situations as our train datasets, it recognized the ramens' labels. However if we put the items in quite different angle, it was hard to recognize. We should have took pictures in more various situations.


# Conclusion

Our project was meaningful that we created our own image data and used it.
1. image resize
2. 오버피팅을 막기위한 여러 방법
3. transfer learning
4. 이미지는 갯수도 많아야하지만 다양해야한다.
