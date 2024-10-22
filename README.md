
## Face Detection Using Viola Jones Haar_cascade Classifiers

Face detection is an important task in computer vision and is used in a wide range of applications such as security systems, photo tagging, and video analytics. One of the most widely used methods for face detection is the Haar Cascade Classifier, which was introduced by Viola and Jones in their seminal paper "Rapid Object Detection using a Boosted Cascade of Simple Features".

![App Screenshot](https://docs.opencv.org/3.4/haar_features.jpg)

In this Repo, we will explore the basic concepts of Haar Cascade Classifiers and how they can be used for face detection. We will also provide a small example using Open CV to detect faces in an image.


## What are Haar Cascade Classifiers?

Haar Cascade Classifiers are based on Haar wavelets, which are mathematical functions that can be used to detect edges in an image. These functions are applied to sub-windows of an image to extract features such as edges, lines, and corners. These features are then used to classify the sub-windows as either containing an object of interest (e.g., a face) or not.


A Haar Cascade Classifier is a machine learning algorithm that uses a set of positive and negative training samples to learn a model for face detection. The classifier is trained using a process called Adaboost, which combines weak classifiers (i.e., classifiers that perform only slightly better than chance) into a strong classifier. The resulting classifier is a weighted combination of weak classifiers that can be used to classify new images.

## Testing Image:
![App Screenshot](https://raw.githubusercontent.com/saikamal3344/Face-Detection-using-Harcascade-Classifiers-/main/girl.png)

## How do Haar Cascade Classifiers work?
Haar Cascade Classifiers work by sliding a sub-window over the image and extracting a set of features for each sub-window. The features are calculated using Haar wavelets and can be represented as a vector. The classifier then uses the vector of features to predict whether the sub-window contains a face or not.
The process of detecting a face using a Haar Cascade Classifier can be summarized as follows:
- Sliding a sub-window over the image
- Extracting a set of features for each sub-window using Haar wavelets
-  Classifying the sub-window as either containing a face or not using the learned model

The sub-windows are typically scaled to different sizes to account for variations in the size of faces in the image.

### Result:

![App Screenshot](https://raw.githubusercontent.com/saikamal3344/Face-Detection-using-Harcascade-Classifiers-/main/re.png)


## 🛠 Skills

        1.Python 
        2.Machine learning 
        3.Statistics
        4.Mathematics
        5.Numpy 
        6.Neural Networks
        7.Deep Learning Multilayer Perceptron concepts 
        8.Computer Vision
    


## Support

For support, email saikamal9797@gmail.com .


## Acknowledgements

 - [Documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)


## 🔗 Links

For my work you can follow:


[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sai-kamal-korlakunta-a81326163/)

[Medium](https://medium.com/@korlakuntasaikamal10)

