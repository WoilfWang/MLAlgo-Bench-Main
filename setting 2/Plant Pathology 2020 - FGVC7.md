You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Plant_Pathology_2020_-_FGVC7_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
#### Problem Statement
Misdiagnosis of the many diseases impacting agricultural crops can lead to misuse of chemicals leading to the emergence of resistant pathogen strains, increased input costs, and more outbreaks with significant economic loss and environmental impacts. Current disease diagnosis based on human scouting is time-consuming and expensive, and although computer-vision based models have the promise to increase efficiency, the great variance in symptoms due to age of infected tissues, genetic variations, and light conditions within trees decreases the accuracy of detection. 
#### Specific Objectives
Objectives of ‘Plant Pathology Challenge’ are to train a model using images of training dataset to 1) Accurately classify a given image from testing dataset into different diseased category or a healthy leaf; 2) Accurately distinguish between many diseases, sometimes more than one on a single leaf; 3) Deal with rare classes and novel symptoms; 4) Address depth perception—angle, light, shade, physiological age of the leaf; and 5) Incorporate expert knowledge in identification, annotation, quantification, and guiding computer vision to search for relevant features during learning. 
#### Resources
Details and background information on the dataset and Kaggle competition ‘Plant Pathology 2020 Challenge’ were published. If you use the dataset for your project, please cite the following peer-reviewed research article
Thapa, Ranjita; Zhang, Kai; Snavely, Noah; Belongie, Serge; Khan, Awais. The Plant Pathology Challenge 2020 data set to classify foliar disease of apples. Applications in Plant Sciences, 8 (9), 2020.


##  Evaluation Metric:
Submissions are evaluated on mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column. 
Submission File
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

    image_id,
    test_0,0.25,0.25,0.25,0.25
    test_1,0.25,0.25,0.25,0.25
    test_2,0.25,0.25,0.25,0.25
    etc.

##  Dataset Description:
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease. 
#### Files
train.csv

    image_id: the foreign key
    combinations: one of the target labels
    healthy: one of the target labels
    rust: one of the target labels
    scab: one of the target labels

images

A folder containing the train and test images, in jpg format.

test.csv

    image_id: the foreign key

sample_submission.csv

    image_id: the foreign key
    combinations: one of the target labels
    healthy: one of the target labels
    rust: one of the target labels
    scab: one of the target labels

train.csv - column name: image_id, healthy, multiple_diseases, rust, scab
test.csv - column name: image_id


## Dataset folder Location: 
../../kaggle-data/plant-pathology-2020-fgvc7. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv, images

## Solution Description:
3.1 Background Introduction

Competition Dataset

The competition dataset includes 1,821 training images and 1,821 test images. Each image belongs to one of four possible labels (healthy, rust disease, scab disease, or a combination of both diseases), with the proportions distributed as 6:1:6:6. This represents a class imbalance problem, and the dataset contains a portion of noisy labels. To address the challenges of small data volume and inaccurate labels, we adopted data augmentation and knowledge distillation strategies.

Evaluation Metric

The evaluation metric is the mean column-wise ROC AUC, which is the average ROC AUC across all classes and measures the model’s performance.

3.2 Data Preprocessing

Data Augmentation

Given the small size of the competition dataset, directly training a model on the raw data may result in overfitting. To enhance model robustness, a series of data augmentation techniques were applied to enrich the training dataset:
	•	Random Brightness Adjustment
	•	Random Contrast Adjustment
	•	Vertical Flip
	•	Horizontal Flip
	•	Random Rotation and Scaling

Additionally, other subtle augmentations like Gaussian blur were employed to further diversify the dataset, enabling the model to learn richer features for improved generalization.

Example code for augmentation using the Albumentations library:

from albumentations import (
    Compose,
    Resize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    MotionBlur,
    MedianBlur,
    GaussianBlur,
    VerticalFlip,
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
)
train_transform = Compose(
    [
        Resize(height=image_size[0], width=image_size[1]),
        OneOf([RandomBrightness(limit=0.1, p=1), RandomContrast(limit=0.1, p=1)]),
        OneOf([MotionBlur(blur_limit=3), MedianBlur(blur_limit=3), GaussianBlur(blur_limit=3),], p=0.5,),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.2,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1,
        ),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ]
)

3.3 Model Selection

We utilized the SE-ResNeXt50 model architecture. The “SE” prefix refers to the Squeeze-and-Excitation mechanism, which works by enhancing important features while suppressing irrelevant ones, akin to attention mechanisms. This method strengthens the extracted features, enabling the model to better recognize fine-grained visual patterns crucial for Fine-Grained Visual Categorization (FGVC) tasks.

3.4 Training Strategy

The training employed the Adam optimizer combined with a cyclic learning rate strategy. This approach reduces overfitting risks while eliminating the need for extensive hyperparameter tuning, making it a highly practical method.

3.5 Error Analysis

Error analysis is a critical step in refining deep learning models. We used heatmaps to visualize the regions the model focused on when predicting each class. By examining misclassified samples, we identified potential improvements in data augmentation and training strategies.

3.6 Self-Distillation

Due to the inherent difficulty in distinguishing between certain disease classes and the presence of noisy labels, the model risks being misled. To mitigate this, we implemented a self-distillation technique:
	1.	Train a five-fold model.
	2.	Combine the validation sets from each fold to create an out-of-fold (OOF) file.
	3.	Blend the OOF predictions with the ground truth labels in a 3:7 ratio.
	4.	Use these softened labels to train a new model.

This strategy assigns probabilistic labels to reduce the impact of noisy labels and simplifies the learning process.

3.7 Model Inference

For the final submission, we applied Test-Time Augmentation (TTA). Predictions were averaged across multiple augmented versions of the test images, leading to improved performance.

This comprehensive approach—including robust data augmentation, knowledge distillation, and test-time techniques—enhanced the model’s accuracy and reliability.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: