You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named 2018_Data_Science_Bowl_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description

**Spot Nuclei. Speed Cures.**

Imagine speeding up research for almost every disease, from lung cancer and heart disease to rare disorders. The 2018 Data Science Bowl offers our most ambitious mission yet: create an algorithm to automate nucleus detection.

We’ve all seen people suffer from diseases like cancer, heart disease, chronic obstructive pulmonary disease, Alzheimer’s, and diabetes. Many have seen their loved ones pass away. Think how many lives would be transformed if cures came faster.

By automating nucleus detection, you could help unlock cures faster—from rare disorders to the common cold. Want a snapshot about the 2018 Data Science Bowl? View this video.

**Why nuclei?**

Identifying the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

By participating, teams will work to automate the process of identifying nuclei, which will  allow for more efficient drug testing, shortening the 10 years it takes for each new drug to come to market. Check out this video overview to find out more.


##  Evaluation Metric:

This competition is evaluated on the mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$IoU(A,B) = \frac{A \cap B}{ A \cup B}.$$


The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95). In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{TP(t)}{TP(t) + FP(t) + FN(t)}.$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{|thresholds|} \sum_t \frac{TP(t)}{TP(t) + FP(t) + FN(t)}.$$

Lastly, the score returned by the competition metric is the mean taken over the individual average precisions of each image in the test dataset.
Submission File
In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).
The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed 
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.
The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.
The file should contain a header and have the following format. Each row in your submission represents a single predicted nucleus segmentation for the given ImageId.

    ImageId,EncodedPixels  
    0114f484a16c152baa2d82fdd43740880a762c93f436c8988ac461c5c9dbe7d5,1 1  
    0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,1 1  
    0999dab07b11bc85fb8464fc36c947fbd8b5d6ec49817361cb780659ca805eac,2 3 8 9  
    etc...
    content_copy

Submission files may take several minutes to process due to the size.

##  Dataset Description:
This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). The dataset is designed to challenge an algorithm's ability to generalize across these variations.
Each image is represented by an associated ImageId. Files belonging to an image are contained in a folder with this ImageId. Within this folder are two subfolders:

    images contains the image file.
    masks contains the segmented masks of each nucleus. This folder is only included in the training set. Each mask contains one nucleus. Masks are not allowed to overlap (no pixel belongs to two masks).

The second stage dataset will contain images from unseen experimental conditions. To deter hand labeling, it will also contain images that are ignored in scoring. The metric used to score this competition requires that your submissions are in run-length encoded format. Please see the evaluation page for details.
As with any human-annotated dataset, you may find various forms of errors in the data. You may manually correct errors you find in the training set. The dataset will not be updated/re-released unless it is determined that there are a large number of systematic errors. The masks of the stage 1 test set will be released with the release of the stage 2 test set.
File descriptions

    /stage1_train/* - training set images (images and annotated masks)
    /stage1_test/* - stage 1 test set images (images only, you are predicting the masks)
    /stage2_test/* (released later) - stage 2 test set images (images only, you are predicting the masks)
    stage1_sample_submission.csv - a submission file containing the ImageIds for which you must predict during stage 1
    stage2_sample_submission.csv (released later) - a submission file containing the ImageIds for which you must predict during stage 2
    stage1_train_labels.csv - a file showing the run-length encoded representation of the training images. This is provided as a convenience and is redundant with the mask image files.

stage1_train_labels.csv - column name: ImageId, EncodedPixels


## Dataset folder Location: 
../../kaggle-data/data-science-bowl-2018. In this folder, there are the following files you can use: stage2_sample_submission_final.csv, stage1_sample_submission.csv, stage1_train_labels.csv, stage2_test_final, stage1_test, stage1_train

## Solution Description:
The third place solution, tie with #2 jacobkie achieving 0.614 on the Private Leader-board, is based on a single Mask-RCNN model using as code-base Matterport's Mask-RCNN. You can use TorchVision for the implementation of solution.

### Summary

The main ideas contain two aspects:

1) Strong scaling augmentation, a lot of zooming in and out and aspect ratio changes before taking the 512x512 crops used as inputs to the model during training.

2) Test time augmentation, I used 15 different augmentations at test time with different rotations, scalings, channel color shifts, etc. This takes a loooong time (aprox. 2 days for the stage_2 test set) and a binary dilation post-processing actually gives a very similar score, so I would use the latter if asked now (although it is easy to tell now that we can see the PL scores..)


### Augmentations

In addition to the scaling augmentation mentioned above I used left-right and up-down flips, random 90 degree rotations, random additional rotation on top of those, random channel color shifts

Parameters:

    CodeBase Type-1 and 2
    MEAN_PIXEL [123.7, 116,8, 103,9]
    LEARNING_RATE Start 0.001 and down to 3*10^-5
    LEARNING_SCHEDULE ~120 always "all"
    RPN_ANCHOR_RATIOS [0.5, 1, 2]
    USE_MINI_MASK True
    MINI_MASK_SHAPE (56,56)
    GPU_COUNT 1
    IMAGES_PER_GPU 2
    STEPS_PER_EPOCH 332
    VALIDATION_STEPS 0
    BACKBONE resnet101
    NUM_CLASSES 1+1
    IMAGE_MIN_DIM 512
    IMAGE_MAX_DIM Not used
    IMAGE_PADDING Not used
    RPN_ANCHOR_SCALES 8,16,32,64,128
    RPN_ANCHOR_STRIDE 1
    BACKBONE_STRIDES 4,8,16,32,64
    RPN_TRAIN_ANCHORS_PER_IMAGE 256
    IMAGE_MIN_SCALE Not used
    IMAGE_RESIZE_MODE crop at training, pad64 for inference
    RPN_NMS_THRESHOLD 0.7
    DETECTION_MIN_CONFIDENCE 0.9
    DETECTION_NMS_THRESHOLD 0.2
    TRAIN_ROIS_PER_IMAGE 600
    DETECTION_MAX_INSTANCES 512
    MAX_GT_INSTANCES 256
    init_with coco
    DATA_AUGMENTATION scaling, crop, flip-lr, flip-up, 90 rotation, rotation, channel_shift



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: