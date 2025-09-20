You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Mechanisms_of_Action_(MoA)_Prediction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
The Connectivity Map, a project within the Broad Institute of MIT and Harvard,  the Laboratory for Innovation Science at Harvard (LISH), and the NIH Common Funds Library of Integrated Network-Based Cellular Signatures (LINCS), present this challenge with the goal of advancing drug development through improvements to MoA prediction algorithms.

#### What is the Mechanism of Action (MoA) of a drug? And why is it important? 
In the past, scientists derived drugs from natural products or were inspired by traditional remedies. Very common drugs, such as paracetamol, known in the US as acetaminophen, were put into clinical use decades before the biological mechanisms driving their pharmacological activities were understood. Today, with the advent of more powerful technologies, drug discovery has changed from the serendipitous approaches of the past to a more targeted model based on an understanding of the underlying biological mechanism of a disease. In this new framework, scientists seek to identify a protein target associated with a disease and develop a molecule that can modulate that protein target. As a shorthand to describe the biological activity of a given molecule, scientists assign a label referred to as mechanism-of-action or MoA for short.
#### How do we determine the MoAs of a new drug? 
One approach is to treat a sample of human cells with the drug and then analyze the cellular responses with algorithms that search for similarity to known patterns in large genomic databases, such as libraries of gene expression or cell viability patterns of drugs with known MoAs.

In this competition, you will have access to a unique dataset that combines gene expression and cell viability data. The data is based on a new technology that measures simultaneously (within the same samples) human cells’ responses to drugs in a pool of 100 different cell types (thus solving the problem of identifying ex-ante, which cell types are better suited for a given drug). In addition, you will have access to MoA annotations for more than 5,000 drugs in this dataset.

As is customary, the dataset has been split into testing and training subsets. Hence, your task is to use the training dataset to develop an algorithm that automatically labels each case in the test set as one or more MoA classes. Note that since drugs can have multiple MoA annotations, the task is formally a multi-label classification problem.    

#### How to evaluate the accuracy of a solution? 
Based on the MoA annotations, the accuracy of solutions will be evaluated on the average value of the logarithmic  loss function applied to each drug-MoA annotation pair.

If successful, you’ll help to develop an algorithm to predict a compound’s MoA given its cellular signature, thus helping scientists advance the drug discovery process.    

This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
For every sig_id you will be predicting the probability that the sample had a positive response for each <MoA> target. For N sig_id rows and M <MoA> targets, you will be making $N\times M$ predictions. Submissions are scored by the log loss:
$$\text{score} = - \frac{1}{M}\sum_{m=1}^{M} \frac{1}{N} \sum_{i=1}^{N} \left[ y_{i,m} \log(\hat{y}_{i,m}) + (1 - y_{i,m}) \log(1 - \hat{y}_{i,m})\right]$$
where:

    N is the number of sig_id observations in the test data (i=1, 2, ..., N)
    M is the number of scored MoA targets (𝑚=1,…,𝑀m=1,…,M)
    $\hat{y}_{i,m}$ is the predicted probability of a positive MoA response for a sig_id
    $y_{i,m}$ is the ground truth, 1 for a positive response, 0 otherwise
    log() is the natural (base e) logarithm

Note: the actual submitted predicted probabilities are replaced with 𝑚𝑎𝑥(𝑚𝑖𝑛(𝑝,1−10−15),10−15)max(min(p,1−10−15),10−15). A smaller log loss is better.

Submission File

You must predict a probability of a positive target for each sig_id-<MoA> pair. The id used for the submission is created by concatenating the sig_id with the MoA target for which you are predicting. The file should have a header and be in the following format:

    sig_id,11-beta-hsd1_inhibitor,ace_inhibitor,...,wnt_inhibitor
    id_000644bb2,0.32,0.01,...,0.57
    id_000a6266a,0.88,0.27,...,0.42
    etc...


##  Dataset Description:
In this competition, you will be predicting multiple targets of the Mechanism of Action (MoA) response(s) of different samples (sig_id), given various inputs such as gene expression data and cell viability data.

Two notes:

    the training data has an additional (optional) set of MoA labels that are not included in the test data and not used for scoring.
    the re-run dataset has approximately 4x the number of examples seen in the Public test.

Files

    train_features.csv - Features for the training set. Features g- signify gene expression data, and c- signify cell viability data. cp_type indicates samples treated with a compound (cp_vehicle) or with a control perturbation (ctrl_vehicle); control perturbations have no MoAs; cp_time and cp_dose indicate treatment duration (24, 48, 72 hours) and dose (high or low). 
    train_drug.csv - This file contains an anonymous drug_id for the training set only.
    train_targets_scored.csv - The binary MoA targets that are scored. 
    train_targets_nonscored.csv - Additional (optional) binary MoA responses for the training data. These are not predicted nor scored.
    test_features.csv - Features for the test data. You must predict the probability of each scored MoA for each row in the test data.
    sample_submission.csv - A submission file in the correct format.

train_drug.csv - column name: sig_id, drug_id


## Dataset folder Location: 
../../kaggle-data/lish-moa. In this folder, there are the following files you can use: test_features.csv, train_targets_nonscored.csv, train_drug.csv, sample_submission.csv, train_targets_scored.csv, train_features.csv

## Solution Description:
### 1. Overview
First of all, many thanks to Kaggle team and Laboratory for Innovation Science at Harvard for providing priceless dataset and hosting this great challenge. This is the my first kaggle competition, I have to say I learned a lot from the valuable community.

This approach consists of 3 single modes:

    1D-CNN
    TabNet
    DNN

and the final submission is generated by weighted average of single models outputs.

The table below lists the performance of the single modes and the final blending in this scheme. The most important part of this scheme may be 1D-CNN 0.01601 (private lb). The final blending only improves by 0.00002 on this basis (private lb 0.01599).

### 2. Pre-processing
#### Sample Normalization
There are some assumptions (may be wrong):

    The raw biological data are often not comparable due to the inevitable existence of systematic errors(e.g., data cannot be produced on the same machine, experiments cannot be performed by the same person)
    Even with totally different treatments, the number of significantly different features won`t be too large.
    It can be simply assume that the distributions of most samples are similar.

Based on these assumptions, samples weres normalized by the distribution of gene expression and cell viability separately.

In short, the gene data in each sample is subtracted from the mean value of 25% and 75% quantiles .

Similarly, the cell data is subtracted from the mean of 25% and 72% quantiles, and then divided by the 4 + 75% quantiles of the new distribution. The slightly different steps for genes and cells are determined by the normality of the distribution after treatment.

Generally, samples normalization should be implemented before features conversion.

Feature Transformation  
After sample normalization, quantile transformation is applied on numerical features.

### 3. Feature Engineering
There is nothing special in feature engineering, even no variance filtering in 1D-CNN model. Different feature processing methods in the other two models are only used to increase the diversity.

    PCA feature are used for all models with different n_components (50 genes + 15 cells for 1D-CNN, 600 genes + 50 cells for TabNet and DNN).
    Statistical features (such as sum, mean) and combination features are used in Tabnet.
    Variance fillter is used in Tabnet and DNN.
    In addition, dummy variable (cp_time, cp_dose) are removed in all models.

### 4. Loss
To deal with the imbalance of targets and reduce overfitting to specific ones, BECloss with label smooth and pos_weight was used in all single models.

As mentioned by others, label smoothing performed well by reducing the overconfident probability .

In addition, targets weight is slightly adjusted by the reciprocal of class frequency.

Specifically, for each target i, the weight is set to log ( Fmin + 100 ) / log ( Fi + 100 ), where the Fi is the number of positive samples in target i, and Fmin donates the min positive count of all target. The constant term 100 is added to prevent the target with low frequency (e.g., with 1 positive sample) from affecting model extremely, and log conversion is to make it smooth.

### 5. Modelling
#### 1) 1d-cnn
This single mode achieves the best performance in this approach ( private score : 0.01601 ). Using such a structure in tabular data is based on the idea that:

    CNN structure performs well in feature extraction, but it is rarely used in tabular data because the correct features ordering is unknown.
    A simple idea is to reshape the data directly into a multi-channel image format, and the correct sorting can be learned by using FC layer through back propagation.

##### Model Architecture
Based on these ideas, the model which extracts features through 1D-CNN (performs better than 2D and 3D in experiments) are implemented. The figure below shows the main structure.

The the architecture of a neural network is described as follows:
	1.	Input (1x937): The network starts with an input feature vector of size ￼.
	2.	Dense Layer (1x4096): The input is passed through a fully connected dense layer, producing an output of size ￼.
	3.	Reshape (256@16x1): The dense output is reshaped into a 3D tensor with dimensions ￼.
	4.	Convolution (512@16x1): A convolutional layer is applied, increasing the depth to 512 while preserving the spatial dimensions ￼.
	5.	Average Pooling (512@8x1): Average pooling is applied, reducing the spatial dimensions by half to ￼ while maintaining the depth at 512.
	6.	Convolution (512@8x1): Another convolutional layer is applied, keeping the dimensions the same (￼).
	7.	Convolution (512@8x1): A second convolutional layer maintains the same dimensions (￼).
	8.	Convolution (512@8x1): A third convolutional layer also retains the dimensions (￼).
	9.	Convolution (512@4x1): The spatial dimensions are reduced to ￼ using another convolutional layer, with the depth remaining at 512.
	10.	Max Pooling + Skip Connection: A max-pooling operation is applied to the previous layer’s output, which is then summed with a skip connection from a preceding layer.
	11.	Flatten (1x2048): The resulting tensor is flattened into a 1D vector of size ￼.
	12.	Dense Layer (1x206): The flattened vector is passed through a final dense layer, producing an output of size ￼.

As shown above, feature dimension is increased through a FC layer firstly. The role of this layer includes providing enough pixels for the image by increasing the dimension, and making the generated image meaningful by features sorting.

Next, data is directly reshaped into image format (size 16*1, chanel 256).

Like the basic CNN model, features are extracted in the next several 1D-Conv layers with a shortcut-like connection.

Finally, the extracted features are used to predict targets through a FC layer after flatten.

#### 2) TabNet
You can directly use TabNet by this line:

    from pytorch_tabnet.tab_model import TabNetRegressor

#### 3) DNN
The described model is a multi-layer feedforward neural network implemented in PyTorch. Here’s a summary of its architecture:
	1.	Input Layer:
	•	Accepts input features (num_features) and applies batch normalization to standardize the data.
	2.	Hidden Layers:
	•	Consists of four fully connected layers with decreasing sizes: [1500, 1250, 1000, 750].
	•	Each layer includes:
	•	Batch normalization to stabilize and accelerate training.
	•	Dropout with progressively decreasing rates ([0.5, 0.35, 0.3, 0.25]) to prevent overfitting.
	•	Leaky ReLU activation for introducing non-linearity.
	3.	Output Layer:
	•	A fully connected layer maps the final hidden layer to the target dimension (num_targets).
	•	Applies weight normalization to improve training stability.
	4.	Forward Propagation:
	•	Input passes sequentially through the layers, where each hidden layer standardizes, regularizes, activates, and transforms the input before feeding it to the next layer.
	•	The output layer produces the final prediction without an explicit activation function.

### 6. Blending
3 single models are blended as the final submission. 

1D-CNN * 0.65 + TabNet * 0.1 + DNN * 0.25

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: