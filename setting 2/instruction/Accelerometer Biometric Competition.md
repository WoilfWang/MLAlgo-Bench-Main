You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Accelerometer_Biometric_Competition_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Since everyone moves differently and accelerometers are fast becoming ubiquitous, this competition is designed to investigate the feasibility of using accelerometer data as a biometric for identifying users of mobile devices.
Seal has collected accelerometer data from several hundred users over a period of several months during normal device usage. To collect the data, we published an app on Googles’s Android PlayStore that samples accelerometer data in the background and posts it to a central database for analysis.
We have uploaded approximately 60 million unique samples of accelerometer data collected from 387 different devices. These are split into equal sets for training and test. Samples in the training set are labeled with the unique device from which the data was collected. The test set is demarcated into 90k sequences of consecutive samples from one device.  A file of test questions is provided in which you are asked to determine whether the accelerometer data came from the proposed device.

##  Evaluation Metric:
Submissions are judged on area under the ROC curve. 

In Matlab (using the stats toolbox):

    [~, ~, ~, auc ] = perfcurve(true_labels, predictions, 1);

In R (using the verification package):

    auc = roc.area(true_labels, predictions)

In python (using the metrics module of scikit-learn):

    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr,tpr)
    
### Submission Format
Each line of your submission should contain an QuestionId and a prediction, IsTrue. Note that you may submit any real-valued number as a prediction,  since AUC is only sensitive to the ranking. sampleSubmission.csv shows a representative valid submission.  The format looks like this:

    QuestionId,IsTrue
    1,0
    2,0.3
    3,99999
    4,-0.8
    etc...

##  Dataset Description:
Your task is to determine whether the accelerometer recordings in the test set (test.csv)  belong to the proposed devices in the question set (questions.csv). 

The following files are provided:

| File name | Description |
|----------|----------|
| train.zip | 30m samples for training, labeled with DeviceId | 
| test.zip | 30m samples for test, split into 90k sequences, labeled with SequenceId |
| questions.csv | 90k questions that match test sequences to DeviceId |


**train.csv**



| Field name | Description |
|----------|----------|
|T|Unix time (miliseconds since 1/1/1970)|
|X|Acceleration measured in g on x co-ordin|
|Y|Acceleration measured in g on y co-ordinate|
|Z|Acceleration measured in g on z co-ordinate|
|DeviceId|Unique Id of the device that generated the samples|

**test.csv**


| Field name | Description |
|----------|----------|
|T|Unix time (miliseconds since 1/1/1970)|
|X|Acceleration measured in g on x co-ordin|
|Y|Acceleration measured in g on y co-ordinate|
|Z|Acceleration measured in g on z co-ordinate|
|SequenceId|Unique sequence number assigned to each question. Each group of samples is labeled with a unique SequenceId. Each SequenceId is matched to a professed DeviceId in the questions.csv file.|

**questions.csv**


| Field name | Description |
|----------|----------|
|QuestionId|Id of question|
|SequenceId|Unique number assigned to each sequence of samples|
|QuizDevice|Professed device that generated the sequence of
accelerometer data in the test file


### Data Preparation

After removing from the data set samples from devices with fewer than 6000 samples collected and sequences of samples during periods of rest exceeding 10 seconds, 60 million samples from 387 users were made available for the competition.
Samples from each device were split chronologically into train and test sets, with the earliest samples assigned to the train set.
The test set was demarcated into 90,000 consecutive sequences of 300 samples each from a same device and each sequence assigned a unique Sequence Id.
Each sequence Id was associated with a professed device Id such that in some cases a true device id was assigned and in others a false device id was assigned.

train.csv - column name: T, X, Y, Z, DeviceId
questions.csv - column name: QuestionId, SequenceId, QuizDevice
test.csv - column name: T, X, Y, Z, SequenceId


## Dataset folder Location: 
../../kaggle-data/accelerometer-biometric-competition. In this folder, there are the following files you can use: train.csv, sampleSubmission.csv, questions.csv, test.csv

## Solution Description:
The solution is proposed by José H. Solórzano, second position in the private Leaderboard.

### Summary
Our top entry in the private leaderboard was produced by a 5-fold cross-validated non-linear blend of
24 individual models, which may be divided into chain-based models, conventional (single sequence)
leak-based models, conventional non-leak models, and chain-based residual models. The most accurate
individual models are those that attempt to build probable “chains” of test sequences, so they can be
classified in groups. The professed device that labels test sequences proved to be a very strong
predictor. This is true when looking at test sequences individually, due to the data preparation
methodology, but more so when sequence chains are considered.

### Modeling Techniques and Training
### Chaining
Test sequences are sequential, as noted in the Data Preparation section of the competition's data page.
In order to determine if sequence B is likely to follow A, a classifier for a pair of sequences (A,B) can
be built. Useful features for this classification task include:

- The timestamp gap between the last sample of sequence A and the first sample of sequence B. A
granular distribution of the gap proved useful.
- The timestamp gap between the first and second samples of sequence B. The distribution of this
feature appears useful in determining if a wide gap (> 230ms) between A and B is plausible.
- The distance between the XYZ position of the last sample of sequence A and the first sample of
sequence B.

We built a non-linear classifier that specializes in pairs of sequences with gaps of at most 10,000 ms. In
this domain, the pair classifier achieves a validation AUC of approximately 0.9997.
A simplified version of the chaining algorithm proceeds as follows:

1. Sort the list of test sequences by first timestamp.
2. Create an empty chain. Pick the first sequence A not already in any chain, and add it to the new
chain.
3. Look forward up to 10,000 ms from A for candidate sequences Bi. Apply the consecutiveness
classifier to each (A,Bi).
4. Pick the candidate Bi that maximizes the classification score. If the maximum score is below a
threshold, then we go back to step 2. Otherwise, we add the most suitable Bi to the current
chain, and we repeat step 3 with A = Bi.

The chaining algorithm implemented also takes advantage of the discrete nature of accelerometer
samples in order to determine if it's at all plausible that a pair of sequences belong together.

Additionally, a running binomial distribution on professed device frequencies can be used to determine
if the chain builder has encountered a likely error at some point while building a chain. Specifically, if
no professed device labels half the sequences, within about 3 standard errors, then we assume
something went wrong.

We also implemented a stochastic chain builder that starts out with two sorted lists of sequences: by
first timestamp and last timestamp. This chain builder can pick any sequence at random in order to
build chains forward and/or backward.

With this methodology, the chain builder creates around 5,000 chains on the competition's test data,
with an average length around 19.

Once a probable sequence chain table is built, classification of test sequences can proceed by applying
a binomial distribution to determine the probability that a true professed device would label sequences
in a chain a given number of times. The binomial distribution is given by:

$$f(k;n,p)=P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$$

In our case, n is the number of sequences in the chain, k is the number of sequences labeled by
professed device D, and p is 0.5. We use logarithms of binomial probability densities as features for
classification. It was helpful to look at the binomial distribution for the professed device that labels the
pertinent sequence, but also that of other professed devices in the chain.

In one of the chain-based models, we used a Bayesian approach, that is, we not only determined the
binomial density for the professed device assuming it's the true device, but also estimated what it would
be if it were a false professed device. For the latter, some assumptions have to be made in order to
estimate p, and because of its uncertainty, is weighted separately by the classifier.

### Chain Residual Models
In chains with only two professed devices, the binomial densities of each of the devices will be
identical, so such chains are not at all useful, when relying on professed devices only. Chains with only
3 professed devices can also pose a problem.

In such scenarios (within a threshold) we use a conventional classifier on a chain, testing each of the
professed devices that label it. Classifying sequences in groups is considerably more accurate than
classifying them individually (~ 0.96 AUC in average for our best conventional classifier). We call
these models “residual” as they only apply to sequences that can't be accurately labeled by the
professed-device-based methodology.

We also built “residual” models that result from uniting short chains with others, but this yielded only
marginal improvements.

### One-vs-all Classification
We created a number of conventional models by using one-vs-all classification. Accelerometer samples
or sets of samples from a device can be transformed into positive training examples consisting of
continuous variables. Negative training examples come from other devices at random. The resulting
classifier can be applied to examples similarly obtained from test sequences. The following are some of
the transformations we attempted:

- XYZ.- This is the simplest transformation, where accelerometer observations from one device
sample are converted to a training example consisting of 3 variables. (Leakage note: Given the
discrete nature of accelerometer samples, a granular distribution classifier would probably 
exploit a leak).
- FFT/DCT.- For a window of samples of a given length we compute a Fast Cosine Transform
(FCT), and use a fraction of the resulting array as variables. This can also be done with FFT and
other transforms. (Leakage status unknown).
- Maximum Displacement.- Starting at a given sample, and looking ahead at most N samples,
we find the one that is farthest away from the original sample. We use both the original point
and the farthest point to come up with 6 variables. (Leakage note: It probably picks up the same
leak as XYZ).
- Segment Features.- Within a window of a given length, we extract a number of features, such
as mean observation, standard deviation, minimum and maximum. In total, we obtain 16
features per segment. (Leakage note: It probably picks up the same leak as XYZ, but maybe to a
lesser extent).
- Interval Variables.- Starting at a given sample, we produce a training example consisting of 3
variables which are the next 3 timestamp gaps. (Leakage note: As this classifier performs much
better in one-vs-all classification than in the leaderboard, it's clear that the distribution of
timestamps is predictive of device type).

### Professed device on single sequences
When sequences are considered individually, the professed device label is still predictive. A first
approximation is to use the number of device samples as a feature, but this is imprecise. Applying a
Bayesian approach yields more accurate results. In short, the probability that a professed device D in
the questions file is a true professed device can be estimated as follows, where n is the expected
number of sequences belonging to D in the test data, and m is the number of times D appears as a
professed device in the questions file.

$$ p= \frac{n}{2m-n}$$

### Discrete nature of accelerometer samples
Given that accelerometer samples are not truly continuous, but discrete, it's possible to determine if a
test sequence can belong to a device, by obtaining an intersection of sample sets from the sequence and
the device. Even though test data is prepared in such a way that false professed devices are largely of
the same device type as the ground-truth device, this is not always the case, so a classifier based on the
discrete nature of samples is somewhat accurate on the leaderboard (~ 0.80 AUC). Presumably, there
are reasons other than device-type matching as to why it works

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: