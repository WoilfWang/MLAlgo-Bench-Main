You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named iMaterialist_Challenge_(Fashion)_at_FGVC5_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
As shoppers move online, it would be a dream come true to have products in photos classified automatically. But, automatic product recognition is tough because for the same product, a picture can be taken in different lighting, angles, backgrounds, and levels of occlusion. Meanwhile different fine-grained categories may look very similar, for example, royal blue vs turquoise in color. Many of today’s general-purpose recognition machines simply cannot perceive such subtle differences between photos, yet these differences could be important for shopping decisions.

Tackling issues like this is why the Conference on Computer Vision and Pattern Recognition (CVPR) has put together a workshop specifically for data scientists focused on fine-grained visual categorization called the FGVC5 workshop. As part of this workshop, CVPR is partnering with Google, Wish, and Malong Technologies to challenge the data science community to help push the state of the art in automatic image classification.

In this competition, FGVC workshop organizers with Wish and Malong Technologies challenge you to develop algorithms that will help with an important step towards automatic product detection – to accurately assign attribute labels for fashion images. Individuals/Teams with top submissions will be invited to present their work live at the FGVC5 workshop.

Kaggle is excited to partner with research groups to push forward the frontier of machine learning. Research competitions make use of Kaggle's platform and experience, but are largely organized by the research group's data science team. Any questions or concerns regarding the competition data, quality, or topic will be addressed by them.

##  Evaluation Metric:
For this competition each image has multiple ground truth labels. We will use Mean F1 score (micro-averaged, see details here) to measure the algorithm quality. The metric is also known as the example based F-score in the multi-label learning literature.

The F1 metric weights recall and precision equally, and a good recognition algorithm will maximize both precision and recall simultaneously. Thus, moderately good performance on both will be favored over extremely good performance on one and poor performance on the other.

Submission File

For every image in the dataset, submission files should contain two columns: image id and predicted labels. Labels should be a space-delimited list. Note that if the algorithm don’t predict anything, the column can be left blank. The file must have a header and should look like the following:

    id,predicted
    12345,0 1 3
    67890,83
    293,
    etc.


##  Dataset Description:
All the data described below are txt files in JSON format.

Overview

    sample_submission_randomlabel.csv: example submission file with random predictions to illustrate the submission file format
    test.json: images of which the participants need to generate predictions. Only image URLs are provided.
    train.json: training data with image urls and labels
    validation.json: validation data with the same format as train.json

Training Data

The training dataset includes images from 228 fashion attribute classes with multiple ground truth labels for each image. It includes a total of 1,014,544 images for training and 10,586 images for validation and 42,590 images for testing.
All train/validation/test sets have the same format as shown below:
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
"label_id" : [int]
}
Note that for each image, we only provide URLs instead of the image content. Users need to download the images by themselves. Note that the image urls are hosted by Wish so they are expected to be stable. 
This year, we omit the names of the labels to avoid hand labeling the test images.
Testing data and submissions
The testing data only has images as shown below:
{
"images" : [image],
}
{
"image_id" : int,
"url" : [string],
}


## Dataset folder Location: 
../../kaggle-data/imaterialist-challenge-fashion-2018. In this folder, there are the following files you can use: test.json, sample_submission.csv, validation.json, train.json

## Solution Description:
Our solution is stacking many CNN models: densenet201, inception_v3, inception_resnet_v2, resnet50, nasnet, xception by another CNN model and a xgboost model. Details below.

#### Single models

We built many CNN models with the same structure: base model -> GlobalMaxPooling2D -> Dense(2048, activation='relu') -> Dropout(0.5) -> Dense(2048, activation='relu') -> Dropout(0.5) -> Dense(NUMBER_OF_CLASSES, activation='sigmoid').

In training time, we applied some augmentations to avoid overfitting: Flip, rotate, AddToHueAndSaturation, GaussianBlur, ContrastNormalization, Sharpen, Emboss, Crop (thanks to imgaug)

In testing time, we applied only one augmentation: flip.

Best single model scores 0.64xx on public LB

p/s: we didn't optimize threshold for f1 score, just using a simple threshold 0.3 for all classes.

#### Stacking
Stacking is the key to this competition. Thanks to Planet: Understanding the Amazon from Space competition and the winner bestfitting. We have read his solution carefully and tried to create our new method.
We found that the correlation between labels is really important. In Planet competition, bestfitting built 17 models to learn the correlation between 17 classes. Should we do that for 228 classes in this competition?
No, we didn't. Instead, we used a single CNN.

We had 9 single models, each model predicted on 9k samples in the validation set. Then we concatenate and reshape each output sample to [number of models, number of classes].
We built a stacking CNN with the structure: Input(shape=(number of models,number of classes,1)) -> Conv2D(filters=8, kernel_size=(3, 1)) -> Conv2D(16, (3, 1)) -> Conv2D(32, (3, 1)) -> Conv2D(64, (3, 1)) -> Flatten() -> Dense(1024) -> Dense(NUMBER_OF_CLASSES, activation='sigmoid').

With a window size (3,1) we hope the CNN can learn the correlation between the prediction of single models. And the last Dense layer can learn the correlation between 228 labels.
Training this model with Kfold (k=5) on the validation set, we can get 0.714x on public LB.

We also tried to use xgboost and MultiOutputRegressor (supported by sklearn).
Training this model with Kfold (k=5) on the validation set, we can get 0.703x on public LB.

Simple weighted CNN model and xgboost model give us final score 0.719x on public LB.




Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: