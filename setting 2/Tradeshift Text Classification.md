You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tradeshift_Text_Classification_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
In the late 90's, Yann LeCun's team pioneered the successful application of machine learning to optical character recognition. 25 years later, machine learning continues to be an invaluable tool for text processing downstream from the OCR process.

Tradeshift has created a dataset with thousands of documents, representing millions of words. In each document, several bounding boxes containing text are selected. For each piece of text, many features are extracted and certain labels are assigned.

In this competition, participants are asked to create and open source an algorithm that correctly predicts the probability that a piece of text belongs to a given class.

##  Evaluation Metric:
Note: due to the size of the submission file for this competition, submission scoring may take up to 5 minutes. This may cause some browsers to hang as they wait for a response. Be patient! You do not need to resubmit. The file will eventually score and be visible on the leaderboard and your submissions page.

Scoring function

The prediction model \\(f\\) for a given set of parameters \\(\theta\\) generates the predicted probabilities \\(\hat{y}_i = \left \langle \hat{y}_{ij} \right \rangle = f_{\theta}(x_i) \in {[0,1]}^K\\) where each element \\(\hat{y}_{ij}\\) is the probability that the jth-label is true for the ith-sample. The goal of the prediction model is that the expected \\(y_{ij}\\) and predicted \\(\hat{y}_{ij}\\) probabilities have similar values. The metric used to score the performance of the prediction model is the negative logarithm of the likelihood function averaged over Nt test samples and K labels.

$$
\textrm{LogLoss} = \frac{1}{N_{t} \cdot K} \sum_{idx=1}^{N_{t} \cdot K} \textrm{LogLoss}_{idx}  \\  = \frac{1}{N_{t} \cdot K} \sum_{idx=1}^{N_{t} \cdot K} \left[ - y_{idx} \log(\hat{y}_{idx}) - (1 - y_{idx}) \log(1 - \hat{y}_{idx})\right]  = \frac{1}{N_{t} \cdot K} \sum_{i=1}^{N_{t}} \sum_{j=1}^K \left[ - y_{ij} \log(\hat{y}_{ij}) - (1 - y_{ij}) \log(1 - \hat{y}_{ij})\right] 
$$

where \\(log()\\) represents the natural logarithm and \\(idx = (i-1) \cdot K + j\\). The inclusion of the logarithm in the metric function highly penalizes predicted probabilities that are confident and wrong. In the worst case, a prediction of true (1) for an expected false (0) sample adds infinity to the \\(\textrm{LogLoss}\\), \\(-log(0)=\infty\\), which makes a total score of infinity regardless the score for the other samples.

This metric is also symmetric in the sense than predicting 0.1 for a false (0) sample has the same penalty as predicting 0.9 for a positive sample (1). The value is bounded between zero and infinity, i.e. \\(\textrm{LogLoss} \in [0, \infty)\\). The competition corresponds to a minimization problem where smaller metric values, \\(\textrm{LogLoss} \sim 0\\), implie better prediction models. 

More information about the ranking of participants can be found in the Rules Page. In order to avoid infinite values and resolution problems, the predicted probabilities \\(\hat{y}_{ij}\\) are bounded within the range \\([10^{-15},1-10^{-15}]\\).

Submission File

The submitted file must contain only one predicted probability per row. For example, the number in the second line is the predicted probability \\(\hat{y}_{1,2}\\) for the 1st-sample and 2nd-label (1_y2), which is calculated from the features \\(x_1\\) and must be similar to the value of \\(y_{1,2}\\).

Besides, the submitted file must contain column and row headers following the format of the sampleSubmission.csv (25MB).

Example

If the testLabels.csv file is (this file is not public):

    id_label,pred
    1_y1,1.0000
    1_y2,0.0000
    1_y3,0.0000
    1_y4,0.0000
    2_y1,0.0000
    2_y2,1.0000
    2_y3,0.0000
    2_y4,1.0000
    3_y1,0.0000
    3_y2,0.0000
    3_y3,1.0000
    3_y4,0.0000
and the testLabelsSubmitted.csv file is (this file is submitted by each participant):

    id_label,pred
    1_y1,0.9000
    1_y2,0.1000
    1_y3,0.0000
    1_y4,0.3000
    2_y1,0.0300
    2_y2,0.7000
    2_y3,0.2000
    2_y4,0.8500
    3_y1,0.1900
    3_y2,0.0000
    3_y3,1.0000
    3_y4,0.2700


##  Dataset Description:
Data extraction

For all the documents, words are detected and combined to form text blocks that may overlap to each other. Each text block is enclosed within a spatial box, which is depicted by a red line in the sketch below. The text blocks from all documents are aggregated in a data set where each text block corresponds to one sample (row).

For example, if we have 3 documents with 34, 62 and 53 text blocks, respectively, the data set will have 149 samples.

Features

For each sample, several features are extracted that are stored in the train.csv and test.csv. The features include content, parsing, spatial and relational information.

    Content: The cryptographic hash of the raw text.
    Parsing: Indicates if the text parses as number, text, alphanumeric, etc.
    Spatial: Indicates the box position, size, etc.
    Relational: Includes information about the surrounding text blocks in the original document. If there is not such a surrounding text block, e.g. a text block in the top of the document does not have any other text block upper than itself, these features are empty (no-value).

The feature values can be:

    Numbers. Continuous/discrete numerical values.
    Boolean. The values include YES (true) or NO (false).
    Categorical. Values within a finite set of possible values.

Labels

The number of samples is \\(N\\), the number of features is \\(M\\) and the number of labels is \\(K\\).

One sample may belong to one or more labels, i.e. multi-label problem. The values in the trainLabels.csv are in the range [0,1], where 0 implies false and 1 implies true. Thus, the ij-element (ith row, jth column) indicates if the i-sample belongs to the j-label, where \\(i \in \left \{ 1, .. , N \right \}\\) and \\(j \in \left \{ 1, .. , K \right \}\\). As a sample may belong to several labels, the sum per row is not always one. In addition, the sum per column does not also add up to one.

Observations

An example of features and labels for the training data is presented below. The dimensions of the example are N=7, M=6 and K=4.

    The order of samples and features is random. In fact, two consecutive samples in the table will most likely not belong to the same document.
    Some documents are OCR'ed; hence, some noise in the data is expected.
    The documents have different formats and the text belongs to several languages.
    The number of pages and text blocks per document is not constant.
    The meaning of the features and class is not provided.


Data dimensions

    Number of samples (N): ~2.1M  (80% training, 20% testing)
    Number of features (M): 145
    Number of labels (K): 33

The test data is split into public (30%) and private (70%) sets, which are used for the public and private leaderboards.

File descriptions

All the files follow a format of comma-separated values (csv) where the headers are 1-indexed. Each row in the files stores a different sample. 

    train.csv - the features \\(x\\) of the training set. Each row corresponds to a different sample, while each column is a different feature.
    trainLabels.csv - the expected labels \\(y\\) for the training set. Each row corresponds to a different sample, while each column is a different label. The order of the rows is aligned with train.csv.
    test.csv - the features \\(x\\) of the test set. Each row corresponds to a different sample, while each column is a different feature.
    sampleSubmission.csv - example of the expected probabilities \\(\hat{y}\\) for the test set. Each row contains two columns, namely one string and the probability of each sample belonging to each label. For example, if the test.csv has 3 samples and 4 labels, the submission file must have 13 rows with these strings in the first column: id_label, 1_y1, 1_y2, 1_y3, 1_y4, 2_y1, 2_y2, 2_y3, 2_y4, 3_y1, 3_y2, 3_y3, 3_y4, 4_y1, 4_y2, 4_y3, 4_y4. More information can be found on the Evaluation page.

trainLabels.csv - column name: id, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30, y31, y32, y33


## Dataset folder Location: 
../../kaggle-data/tradeshift-text-classification. In this folder, there are the following files you can use: trainLabels.csv, train.csv, sampleSubmission.csv, test.csv

## Solution Description:
Here is my solution (6th rank):

I used similar technique with meta-level but with some differences.

My splits were 1000000:700000, 1500000:200000

As you know, if remove all hash and yesno columns, there are 85 features which
which are divided into 5 groups (17 * 5 = 85). So I create additional features X[17] - X[0], X[18] - X[1], etc. And on all numeric features there was RF with criterion entropy on 1600 trees. Add 32 results to the second level.

Linear SVM on all hash features were not good enough for me, so I made Linear SVM separately on each hash feature, transforming TF-IDF before. And there are 2 good hash features, others not. So I add 32*2 = 64 to the second level.

y33 is '1' when others '0'. So I made some trick - add sum of predictions y1...y32 to the second level. So, I add 3 new features (one from each classifier before)

Also I add all raw numerical features to the second level (another 135)

The second level is XGBoost with bagging on objects.

Another cool trick is postprocessing result. I calculated sum of final predictions y1...y32. (sum_y = sum(y1:y32)). Then if sum_y is bigger than 1, I replaced to 1. (sum_y = 1 if sum_y > 1)

And final y33 is the linear combination:

new_y33 = alpha * y33 + (1 - alpha) * (1 - sum_y) with alpha around 0.6

This gave an improvement on any solution.

Another model was RF on second level. But with some trick. You need every predictions of y replace to:

new_y = 0.5 * ((2 * abs(y - 0.5)) ** beta) * sign(y - 0.5) + 0.5 with beta around 0.5

It is very effective to fix predictions from RF.

The final solution is just linear combination of all of it.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: