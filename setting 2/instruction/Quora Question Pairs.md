You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Quora_Question_Pairs_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Where else but Quora can a physicist help a chef with a math problem and get cooking tips in return? Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.

Currently, Quora uses a Random Forest model to identify duplicate questions. In this competition, Kagglers are challenged to tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. Doing so will make it easier to find high quality answers to questions resulting in an improved experience for Quora writers, seekers, and readers.

##  Evaluation Metric:
Submissions are evaluated on the log loss between the predicted values and the ground truth.

#### Submission File
For each ID in the test set, you must predict the probability that the questions are duplicates (a number between 0 and 1). The file should contain a header and have the following format:

    test_id,is_duplicate
    0,0.5
    1,0.4
    2,0.9
    etc.

##  Dataset Description:
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.

Please note: as an anti-cheating measure, Kaggle has supplemented the test set with computer-generated question pairs. Those rows do not come from Quora, and are not counted in the scoring. All of the questions in the training set are genuine examples from Quora.
#### Data fields

    id - the id of a training set question pair
    qid1, qid2 - unique ids of each question (only available in train.csv)
    question1, question2 - the full text of each question
    is_duplicate - the target variable, set to 1 if question1 and question2 have essentially the same meaning, and 0 otherwise.

train.csv - column name: id, qid1, qid2, question1, question2, is_duplicate
test.csv - column name: test_id, question1, question2


## Dataset folder Location: 
../../kaggle-data/quora-question-pairs. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
We can simply divide the solution into different parts: Pre-processing, Feature Engineering, Modeling and Post-processing.

#### Pre-processing
We made some different versions of original data (train.csv & test.csv).

    Text-cleaning: spell correction, symbol processing, acronyms restore, …
    Word-stemming: SnowballStemmer, …
    Shared-word-removing: delete the words appeared in the both sides

#### Feature Engineering
There was around 1400+ features in the Feature Pool which will be combined in different ways. These features can be classified as the following categories.

    Statistic: rate of shared words, length of sentences, number of words, …
    NLP: analysis of grammar tree, negative words count, …
    Graph: pagerank, hits, shortest path, clique size, …

#### Modeling
We used DL Models, XGB, LGB and LR. The best single model scored about 0.122~0.124 on the LB. We build a multi-layer stacking system to ensemble different models together (about 140+ model results), this method can get a gain of ~0.007 on public LB.

#### Post-processing
As we all knonw, the distribution of the training data and test data were quite different. We cutted the data into different parts according to the clique size and rescale the results in different parts, this method can get a gain of ~0.001 on public LB.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: