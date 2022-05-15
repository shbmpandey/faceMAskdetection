                        FACE MASK DETECTION 
 
 

   ABSTRACT
The present scenario of COVID-19 demands an efficient face mask detection application. The main goal of the project is to implement this system at entrances of colleges, airports, hospitals, and offices where chances of spread of COVID-19 through contagion are relatively higher. Reports indicate that wearing face masks while at work clearly reduces the risk of transmission. It is an object detection and classification problem with two different classes (Mask and Without Mask). A hybrid model using deep and classical machine learning for detecting face mask will be presented. A dataset is used to build this face mask detector using Python, OpenCV, and TensorFlow and Keras. While entering the place everyone should scan their face and then enter ensuring they have a mask with them. If anyone is found to be without a face mask, a beep alert will be generated. As all the workplaces are opening. The number of cases of COVID-19 are still getting registered throughout the country. If everyone follows the safety measures, then it can come to an end. Hence to ensure that people wear masks while coming to work we hope this module will help in detecting it.
 
 
 
 
 
 
 
INTRODUCTION 
 
 
A new strain of virus was identified in humans, known as novel coronavirus (nCoV), which had never previously been identified in humans. Coronaviruses (CoV) are a wide group of viruses which cause illness that range from basic colds to infections like Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). The first infected patient of coronavirus was found in December 2019. The habit of wearing face masks while stepping out is rising due to the COVID- 19 coronavirus epidemic. Before Covid-19, masks were worn by people to protect their health from air pollution. Scientists have concluded that wearing face masks works on decreasing COVID-19 transmission. In 2020, the rapid spread of COVID-19 led the World Health Organization to declare COVID- 19 as a global pandemic. The virus spreads through close contact of humans and in crowded/overcrowded places. Among them cleaning hands, maintaining a safe distance, wearing a mask, refraining from touching eyes, nose, and mouth are the main, where wearing a mask is the simplest one. Unfortunately, people are not following these rules properly which is resulting in speeding the spread of this virus. The solution can be to detect the people not wearing mask and informing their authorities. Face mask detection is a technique to find out whether the person is wearing a mask or not. In medical applications Deep learning techniques are highly used as it allows researchers to study and evaluate large quantities of data. Deep learning models have shown a great role in object detection. These models and architectures can be used in detecting the mask on a face. Here we introduce a face mask detection model which is based on computer vision and deep learning.  
 
 
 
LITERATURE REVIEW
 
 Sujatha and Chatterjee [1] proposed a model that could be useful to foresee the spread of COVID2019 by using linear regression, Multilayer perceptron and Vector autoregression model on the COVID-19 kaggle data to envision the epidemiological example of the malady and pace of COVID-2019 cases in India. Navares et al. [2] introduced an answer for the issue of anticipating everyday medical clinic confirmations in Madrid because of circulatory and respiratory cases dependent on biometeorological markers. Cui and Singh created and applied the MRE hypothesis for month to month streamflow prediction with spectral power as a random variable. A system [4] that restricts the growth of COVID-19 by finding out people who are not wearing any facial mask in a smart city network where all the public places are monitored with ClosedCircuit Television (CCTV) cameras. Firstly, CCTV cameras are used to capture real-time video footage of different public places in the city. From that video footage, facial images are extracted and these images are used to identify the mask on the face. Another model [5] for face detection using semantic segmentation in an image by classifying each pixel as face and non-face i.e. effectively creating a binary classifier and then detecting that segmented area. It works very well not only for images having frontal faces but also for non-frontal faces. The most helpful project for us, proposed [6] a method for automatic door access system using face recognition technique by using python programming and from OpenCV library Haar cascade method. Object Detection using Haar feature-based cascade classifiers is an effective object detection method proposed by Paul Viola and Michael Jones [7]. Another research [8] in which a hybrid model using deep and classical machine learning for face mask detection is presented. A face mask detection dataset consists of with mask and without mask images, then using OpenCV to do real-time face detection from a live stream via webcam. Another tutorial [9], had a two-phase COVID-19 face mask detector, detailing how computer vision/deep learning pipeline will be implemented. The trained COVID-19 face mask detector, will implement two more additional Python scripts used to detect COVID-19 face masks in images and detect face masks in real-time video streams. 
 
OBJECTIVES
The main objective of the “Face mask recognition” project is to provide some effective technology for preventing the spread of Coronavirus. Primary objectives behind the development of this system are as follows: -
• Prevent the spread of Coronavirus by promoting the use of face masks with the help of effective technology to detect the face mask.
• Help to take necessary precautions for the safety of society by predicting the future outbreaks of COVID-19.
• Ensure a safe working environment.
• Save the lives of people.
 
 



METHODOLOGY USED

We have devised a smart framework for detecting facemask in this paper. As the cases of covid-19 are decreasing, maximum workplaces are opening with half or full employees. Even the education institutes are planning to be opened. For screening the people not wearing masks, this system can be installed in the entrances of enterprises, educational institutes, public and private offices. If the system detects a person’s face with no mask, it will generate a beep alerting them to wear a mask. The block diagram of the developed framework is depicted in Fig. 1.

A. Proposed workflow

We decided to build a very simple and basic Convolutional Neural Network (CNN) model using TensorFlow with Keras library and OpenCV to detect if you are wearing a face mask to protect yourself. All the aspects of our work are described below.

B. Deep learning architecture

The deep learning architecture learns various important nonlinear features from the given
samples. Then, this learned architecture is used to predict previously unseen samples.



C. Image Processing 
 
Haar Cascade Classifier will detect the input from videocam. The images captured by the system's webcam required pre-processing before going to the next step. In the pre-processing step, the image is transformed into a grayscale image because the RGB colour image contains so much redundant information that is not necessary for face mask detection. Then, we resize the images into (150x150) size to maintain uniformity of the input images to the architecture. Then, the images are normalised and after normalisation, the value of a pixel resides in the range from 0 to 1. Normalisation helped the learning algorithm to learn faster and captured necessary features from the images.
 


D.Dataset Collection

To train our deep learning architecture, we collected images. The architecture of the learning technique highly depends on CNN. Data from source[10] is collected for training and testing the model. Dataset contains images of faces only. It consists of about 1,315 images in which 658 images contain people with face masks and 657 images containing people without face masks. For training purposes, 80% images of each class are used and the rest of the images are utilised for testing purposes. Fig. 2 shows some of the images of two different classes.

E.Architecture Development

The learning model is based on CNN which is very useful for pattern recognition from images. Neural Networks need to see data from both the classes. The network comprises an input layer, several hidden layers and an output layer. The hidden layers consist of multiple convolution layers. The features extracted by CNN are used by multiple dense neural networks for  classification purposes. The architecture contains three pairs of convolution layers each 31 followed by one max pooling layer. The convolution layer contains 100 kernels of window size 3x3. Max pooling layer of window size 2x2. This layer will be aggregating the results from the
previous convolution layer and will be picking the max value in that 2x2 window. It decreases the spatial size of the representation and thereby reduces the number of parameters. As a result, the computation is simplified for the network. The output of the convolution layers will be flattened and will be converted into a 1-D array. Then there is one dropout layer and two dense layers. The dropout layer prevents the network from overfitting by dropping out units. The dense layer comprises a series of neurons each of them learn nonlinear features. The flattened result will be fed to the first dense layer of 50 nodes. Then finally the second dense layer contains two nodes as there are two classes.
 
F.Alert Generation

The purpose of our system is to screen people not wearing face masks. The learning architecture generates results on the input image, classifying the image into mask or no mask classes. If a person is detected not wearing a mask then a beep alert will be generated until the mask is put on. And if everyone is wearing a mask then they will be safe from the virus. In this way our system
would help greatly to limit the growth of COVID-19.

                  
 
 
                                                                                             Fig 1. Training and Loss accuracy 












                                                                                       

                                                                                       Fig 2. Block Diagram of the proposed model



RESULT ANALYSIS

By preserving a reasonable proportion of different classes, the dataset is partitioned into training and testing sets. The dataset comprises of 1315 samples in total where 80% is used in training
phase and 20% is used in the testing phase. The developed architecture is trained for 10 epochs since further training results cause overfitting on the training data. Overfitting generally occurs when a model learns the unwanted patterns of the training samples. Hence, training accuracy increases but test accuracy decreases.
Fig1. shows the graphical view of accuracy and loss respectively. The trained model showed 95% accuracy.


LIMITATIONS AND FURTHER WORKS

The developed system can detect the live video streams but does not keep a record. Unlike the CCTV camera footage the admin can not rewind, play or pause it. As whenever a strict system is imposed people always try to break it. Hence when a person is detected with no mask, the head of the organisation can be notified via mail that so and so person entered without mask. The proposed system can be integrated with databases of respective organisations to keep a record of the person who entered without a mask. With more complex functions a screenshot of the person’s face can also be attached to keep it as a proof.


CONCLUSION

As the technology is booming with emerging trends therefore the novel face mask detector which can possibly contribute to public healthcare. The model is trained on an authentic dataset. We used OpenCV, tensor flow, keras and CNN to detect whether people were wearing face masks or not. The models were tested with images and real-time video. The accuracy of the model is achieved and the optimization of the model is a continuous process and we are building an accurate solution by tuning the hyper parameters. This specific model could be used as a use case for edge analytics. By the developing this system, we can detect if the person is wearing a face mask and allow their entry would be of great help to the society.



REFERENCES

[1] R. Sujatha, Jyotir Chatterjee and Aboul ella Hassanien, “A machine learningmethodology for forecasting of the COVID-19 cases in India,” Apr 18, 2020. [Online].
Available:-https://www.techrxiv.org/articles/preprint/A_machine_learning_methodology_for_forecasting_of_the_COVID-19_cases_in_India/12143685/1.

[2] Ricardo Navares, Julio Díaz, Cristina Linares and José L. Aznarte, “Comparing ARIMA and computational intelligence methods to forecast daily hospital admissions due to circulatory
and respiratory causes in Madrid,” Mar 3, 2018. [Online].
Available: https://link.springer.com/article/10.1007%2Fs00477-
018-1519-z. 

[3] Huijuan Cui and Vijay P. Singh, “Application of minimum relative entropy theory for streamflow forecasting,” Sept 8,2016. [Online].
Available:https://link.springer.com

[4] Mohammad Marufur Rahman, Md. Motaleb Hossen Manik and Md. Milon Islam, 2020 IEEE International IOT, Electronics and Mechatronics Conference (IEMTRONICS), “ An Automated System to Limit COVID-19 Using Facial Mask Detection in Smart City Network,”Oct.2020.[Online].
Available:https://www.researchgate.net/publication/344563082_An_Automated_System_to_Limit_COVID-19_Using_Facial_Mask_Detection_in_Smart_City_Network.

[5] Toshan Meenpal, Ashutosh Balakrishnan and Amit Verma, 2019 4th International Conference on Computing, Communications and Security (ICCCS), “ Facial Mask Detection using Semantic Segmentation,” Oct. 2019,[Online]. Available: (PDF) Facial Mask Detection using Semantic Segmentation
(researchgate.net).

[6] Tejas Saraf, Ketan Shukla, Harish Balkhande and Ajinkya Deshmukh, International
Research Journal of Engineering and Technology (IRJET), “Automated door access control system using face recognition,” Apr 4, 2018. [Online].
Available: https://www.irjet.net/archives/V5/i4/IRJET-V5I4671.pdf.

[7] Paul Viola and Michael Jone, Conference On Computer Vision And Pattern Recognition
2001, “Rapid Object Detection using a Boosted Cascade of Simple Features,” [Online].
Available:
https://www.cs.cmu.edu/.

[8] Vinitha.V and Velantina.V, International Research Journal of Engineering and Technology (IRJET), “Covid-19 facemask detection with deep learning and computer vision,” Aug, 2020.
[Online]. Available: https://www.irjet.net.

[9] Adrian Rosebrock, “COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow,
and Deep Learning,” May 4, 2020. [Online].
Available:https://www.pyimagesearch.com













