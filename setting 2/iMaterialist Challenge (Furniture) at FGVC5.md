You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named iMaterialist_Challenge_(Furniture)_at_FGVC5_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
As shoppers move online, it’d be a dream come true to have products in photos classified automatically. But, automatic product recognition is challenging because for the same product, a picture can be taken in different lighting, angles, backgrounds, and levels of occlusion. Meanwhile different fine-grained categories may look very similar, for example, ball chair vs egg chair for furniture, or dutch oven vs french oven for cookware. Many of today’s general-purpose recognition machines simply can’t perceive such subtle differences between photos, yet these differences could be important for shopping decisions.

Tackling issues like this is why the Conference on Computer Vision and Pattern Recognition (CVPR) has put together a workshop specifically for data scientists focused on fine-grained visual categorization called the FGVC5 workshop. As part of this workshop, CVPR is partnering with Google, Malong Technologies and Wish to challenge the data science community to help push the state of the art in automatic image classification.

In this competition, FGVC5 workshop organizers and Malong Technologies challenge you to develop algorithms that will help with an important step towards automatic product recognition – to accurately assign category labels for furniture and home goods images. Individuals/Teams with top submissions will be invited to present their work live at the FGVC5 workshop.

Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.

##  Evaluation Metric:
For this competition each image has one ground truth label. An algorithm to be evaluated will produce 1 label per image. If the predicted label is the same as the groundtruth label, then the error for that image is 0, otherwise it is 1. The final score is the error averaged across all images. 

Submission File

For each image in the test set, you must predict 1 class label. The csv file should contain a header and have the following format:

    id,predicted
    12345,0
    67890,83
    etc.


##  Dataset Description:
All the data described below are txt files in JSON format.

Overview

    train.json: training data with image urls and labels
    validation.json: validation data with the same format as train.json
    test.json: images of which the participants need to generate predictions. Only image URLs are provided.
    sample_submission_randomlabel.csv: example submission file with random predictions to illustrate the submission file 
    format

Training Data

The training dataset includes images from 128 furniture and home goods classes with one ground truth label for each image. It includes a total of 194,828 images for training and 6,400 images for validation and 12,800 images for testing.
Train and validation sets have the same format as shown below:
{
"images" : [image],
"annotations" : [annotation],
}
image{
"image_id" : int,
"url": [string]
}
annotation{
"image_id" : int,
"label_id" : int
}
Note that for each image, we only provide URL instead of the image content. Users need to download the images by themselves. Note that the image urls may become unavailable over time. Therefore we suggest that the participants start downloading the images as early as possible. We are considering image hosting service in order to handle unavailable URLs. We'll update here if that could be finalized.
This year, we omit the names of the labels to avoid hand labeling the test images.

Testing data and submissions

The testing data only has images as shown below:
{
"images" : [image],
}
image {
"image_id" : int,
"url" : [string],
}
We also provide a sample submission csv file as an example. The evaluation section has a more detailed description of the submission format.

## Dataset folder Location: 
../../kaggle-data/imaterialist-challenge-furniture-2018. In this folder, there are the following files you can use: test.json, validation.json, sample_submission_randomlabel.csv, train.json

## Solution Description:
First of all, we should thank competition sponsors and Kaggle for organizing a competition that is quite interesting. Besides,  all the competitors are quite excellent, maybe we are just a little lucky. Next, we want to briefly share our methods here.

We use pytorch to do all the training and testing, and our pretrained models are almost from the https://github.com/Cadene/pretrained-models.pytorch. Thanks for Cadene's sharing.

Most of our codes can be found here https://github.com/skrypka/imaterialist-furniture-2018,  just as Dowakin shared before, and we have updated it.

We totally have 12688 pictures for testing, for the missing ones, we just use the result in the sample.csv .
The models we choose are inceptionv4, densenet201, densenet161, dpn92, xception, inceptionrResNetv2, resnet152, senet154, renext101 and nasnet. The accuracy in the val set is 85% to 87%，except the nasnet which is quite hard to train and the accuracy is only 84%. 

For training I used SGD in most cases. The learning rate of the fully connected layer is ten times than others. The initial lr is 0.001 and switched to 0.0001 later.But for SEnet, it is hard to train with SGD, so I used Adam first. 
For testing, we use 12TTA = (normal+horizontal flip)6 crops(full image, center, 4 corners) for each model and save the result. Finally we use gmean to get the final result  for the 1210 12688*128 arrays. And the result is greatly improved  in this way. Apart from that, we notice the imbalance of the result. The test set should be 100 pics for each class, but the training set is not imbalanced. Thus the softmax will jugde by the training rate. We wrote a function to calibrate the probablity. You can see https://www.kaggle.com/dowakin/probability-calibration-0-005-to-lb/notebook.
We also add some weights to the 10 results when we do ensembing, which is useful to get a high score on the public leaderbord. However, for  the final submit we removed it just to avoid the overfitting in the 30% datas. It seems to be effective for the final score..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: