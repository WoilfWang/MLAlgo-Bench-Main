You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Dogs_vs._Cats_Redux:_Kernels_Edition_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
In 2013, we hosted one of our favorite for-fun competitions:  Dogs vs. Cats. Much has since changed in the machine learning landscape, particularly in deep learning and image analysis. Back then, a tensor flow was the diffusion of the creamer in a bored mathematician's cup of coffee. Now, even the cucumber farmers are neural netting their way to a bounty.

Much has changed at Kaggle as well. Our online coding environment Kernels didn't exist in 2013, and so it was that we approached sharing by scratching primitive glpyhs on cave walls with sticks and sharp objects. No more. Now, Kernels have taken over as the way to share code on Kaggle. IPython is out and Jupyter Notebook is in. We even have TensorFlow. What more could a data scientist ask for? But seriously, what more? Pull requests welcome.

We are excited to bring back the infamous Dogs vs. Cats classification problem as a playground competition with kernels enabled. Although modern techniques may make light of this once-difficult problem, it is through practice of new techniques on old datasets that we will make light of machine learning's future challenges.

##  Evaluation Metric:
Submissions are scored on the log loss:
$$\textrm{LogLoss} = - \frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)\right],$$
where

    n is the number of images in the test set
    \\( \hat{y}_i \\) is the predicted probability of the image being a dog
    \\( y_i \\) is 1 if the image is a dog, 0 if cat
    \\( log() \\) is the natural (base e) logarithm

A smaller log loss is better.

Submission File

For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

    id,label
    1,0.5
    2,0.5
    3,0.5
    ...

##  Dataset Description:
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog (1 = dog, 0 = cat).

## Dataset folder Location: 
../../kaggle-data/dogs-vs-cats-redux-kernels-edition. In this folder, there are the following files you can use: sample_submission.csv, train, test

## Solution Description:
I spent relatively little time on preprocessing and feature engineering. I had split data for various cross validation folds on disk, in order to ensure the full consistency across multiple models/machines, as well as for easier access by various command line tools that I used. For one of my models I’ve done a lot of image augmentation — cropping, shearing, rotating, flipping, etc.

Just like with most other image recognition/classification problems, I have completely relied on Deep Convolutional Neural Networks (DCNN). I have built a simple convolutional neural network (CNN) in Keras from scratch, but for the most part I’ve relied on out-of-the-box models: VGG16, VGG19, Inception V3, Xception, and various flavors of ResNets. My simple CNN managed to get the score in the 0.2x range on the public leaderboard (PL). My best models that I build using features extracted by applying retrained DCNNs got me into the 0.06x range on PL. Stacking of those models got me in the 0.05x range on PL. My single best fine-tuned DCNN got me to 0.042 on PL, and my final ensemble gave me the 0.35 score on PL. 

I have primarily used Keras and a Facebook implementation of pretrained ResNets. The latter is written in Torch, so as I am not proficient in Lua, I had to develop all sorts of hacks to get the output of command line tools into my main Python scripts. I have also used OpenCV, XGBoost and sklearn for image manipulation and stacking.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: