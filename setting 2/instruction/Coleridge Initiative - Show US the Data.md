You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Coleridge_Initiative_-_Show_US_the_Data_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
This competition challenges data scientists to show how publicly funded data are used to serve science and society. Evidence through data is critical if government is to address the many threats facing society, including;  pandemics, climate change,  Alzheimer’s disease, child hunger, increasing food production, maintaining biodiversity, and addressing many other challenges. Yet much of the information about data necessary to inform evidence and science is locked inside publications. 
Can natural language processing find the hidden-in-plain-sight data citations? Can machine learning find the link between the words used in research articles and the data referenced in the article? 
Now is the time for data scientists to help restore trust in data and evidence. In the United States, federal agencies are now mandated to show how their data are being used. The new Foundations of Evidence-based Policymaking Act requires agencies to modernize their data management. New Presidential Executive Orders are pushing government agencies to make evidence-based decisions based on the best available data and science. And the government is working to respond in an open and transparent way.
This competition will build just such an open and transparent approach. The results will show how public data are being used in science and help the government make wiser, more transparent public investments. It will help move researchers and governments from using ad-hoc methods to automated ways of finding out what datasets are being used to solve problems, what measures are being generated, and which researchers are the experts. Previous competitions have shown that it is possible to develop algorithms to automate the search and discovery of references to data. Now, we want the Kaggle community to develop the best approaches to identify critical datasets used in scientific publications.    
In this competition, you'll use natural language processing (NLP) to automate the discovery of how scientific data are referenced in publications. Utilizing the full text of scientific publications from numerous research areas gathered from CHORUS publisher members and other sources, you'll identify data sets that the publications' authors used in their work.  
If successful, you'll help support evidence in government data. Automated NLP approaches will enable government agencies and researchers to quickly find the information they need. The approach will be used to develop data usage scorecards to better enable agencies to show how their data are used and bring down a critical barrier to the access and use of public data.    
The Coleridge Initiative is a not-for-profit that has been established to use data for social good. One way in which the organization does this is by furthering science through publicly available research.    
Resources
Coleridge Data Examples
Rich Search and Discovery for Research Datasets
Democratizing Our Data
NSF"Rich Context" Video
Acknowledgments
United States Department of Agriculture
United States Department of Commerce
United States Geological Survey
National Oceanic and Atmospheric Administration
National Science Foundation
National Institutes of Health
CHORUS
Westat
Alfred P. Sloan Foundation
Schmidt Futures
Overdeck Family Foundation


This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
The objective of the competition is to identify the mention of datasets within scientific publications. Your predictions will be short excerpts from the publications that appear to note a dataset.
Submissions are evaluated on a Jaccard-based FBeta score between predicted texts and ground truth texts, with Beta = 0.5 (a micro F0.5 score). Multiple predictions are delineated with a pipe (|) character in the submission file.
The following is Python code for calculating the Jaccard score for a single prediction string against a single ground truth string. Note that the overall score for a sample uses Jaccard to compare multiple ground truth and prediction strings that are pipe-delimited - this code does not handle that process or the final micro F-beta calculation.
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
content_copy
Note that ALL ground truth texts have been cleaned for matching purposes using the following code:
def clean_text(txt):
    return re.sub('[^A-Za-z0-9]+', ' ', str(txt).lower())
content_copy
For each publication's set of predictions, a token-based Jaccard score is calculated for each potential prediction / ground truth pair. The prediction with the highest score for a given ground truth is matched with that ground truth.

Predicted strings for each publication are sorted alphabetically and processed in that order. Any scoring ties are resolved on the basis of that sort.
Any matched predictions where the Jaccard score meets or exceeds the threshold of 0.5 are counted as true positives (TP), the remainder as false positives (FP).
Any unmatched predictions are counted as false positives (FP).
Any ground truths with no nearest predictions are counted as false negatives (FN).

All TP, FP and FN across all samples are used to calculate a final micro F0.5 score. (Note that a micro F score does precisely this, creating one pool of TP, FP and FN that is used to calculate a score for the entire set of predictions.) 
Submission File
For each publication Id in the test set, you must predict excerpts (multiple excerpts divided by a pipe character) for PredictionString variable. The file should contain a header and have the following format:
Id,PredictionString
000e04d6-d6ef-442f-b070-4309493221ba,space objects dataset|small objects data
0176e38e-2286-4ea2-914f-0583808a98aa,small objects dataset
01860fa5-2c39-4ea2-9124-74458ae4a4b4,large objects
01e4e08c-ffea-45a7-adde-6a0c0ad755fc,space location data|national space objects|national space dataset
01fea149-a6b8-4b01-8af9-51e02f46f03f,a dataset of large objects
etc.
content_copy

##  Dataset Description:
The objective of the competition is to identify the mention of datasets within scientific publications. Your predictions will be short excerpts from the publications that appear to note a dataset. Predictions that more accurately match the precise words used to identify the dataset within the publication will score higher. Predictions should be cleaned using the clean_text function from the Evaluation page to ensure proper matching.
Publications are provided in JSON format, broken up into sections with section titles.
The goal in this competition is not just to match known dataset strings but to generalize to datasets that have never been seen before using NLP and statistical techniques. A percentage of the public test set publications are drawn from the training set - not all datasets have been identified in train, so these unidentified datasets have been used as a portion of the public test labels. These should serve as guides for the difficult task of labeling the private test set.
Note that the hidden test set has roughly ~8000 publications, many times the size of the public test set. Plan your compute time accordingly.
Files

train - the full text of the training set's publications in JSON format, broken into sections with section titles
test - the full text of the test set's publications in JSON format, broken into sections with section titles
train.csv - labels and metadata for the training set
sample_submission.csv - a sample submission file in the correct format

Columns

id - publication id - note that there are multiple rows for some training documents, indicating multiple mentioned datasets
pub_title - title of the publication (a small number of publications have the same title)
dataset_title - the title of the dataset that is mentioned within the publication
dataset_label - a portion of the text that indicates the dataset
cleaned_label - the dataset_label, as passed through the clean_text function from the Evaluation page

train.csv - column name: Id, pub_title, dataset_title, dataset_label, cleaned_label


## Dataset folder Location: 
../../kaggle-data/coleridgeinitiative-show-us-the-data. In this folder, there are the following files you can use: train.csv, sample_submission.csv, train, test

## Solution Description:
Our solution is composed of 6 parts below.

    LB probing
    Acronym detection
    Acronym detection version 2
    String-matching with dataset-names from external data
    Dataset-name variation detection using NER
    String-matching with dataset-names from the train data

#### 1. LB probing
The metric of this competition is F-score. Under this metric, assuming a current score is F, a newly detected label can improve the score when the expected value that the label is true positive is greater than 0.8F. Therefore, it is important to estimate the private test score in order to determine the detection threshold. For example, if F is 0.6, the best threshold is 0.48 and if F is 0.4, the best threshold is 0.32. For this reason, it is very important to know the number of the training-data labels in the private test data, because it affects the private test score strongly.

To tackle this problem, we did LB probing. In this competition, the public test data contains duplicates of the train data. Therefore, we can create a submission only with true positive labels and with no false positive labels by applying true positive labels of the train data to their duplicates. Thus, by setting the number of true positive labels to a value related to the hidden test data, we can get information about the hidden test data from the submission score. Using this strategy, we got rough estimates of values below (actual codes are this, I shared in the competition period, and this).

    The number of the public test data: 923
    The number of the private test data: 7,695
    The number of labels in the public test data: 8,546
    The number of labels in the private test data: 62,671
    The number of detected labels in the test data by string-matching of train-data labels: 1,717

From these results, we found that the public test score of train-data label string-matching is very high (0.530), but there are very few train-data labels in the test data (1,717). Therefore, at least 1,600 of the train-data labels in the test data might be in the public test data and very few might be in the private test data. Therefore, the public score of submission which discards the train-data labels from the prediction will correlate well with the private test score. The best submission can be obtained by finding the submission with the max score without train-data labels and adding train-data label string-matching to it.

By this approach, we succussed to select the best-private-score submission out of our 201 submissions. We knew we can survive the big shake of the private test LB, this was a very very big advantage for our team.

#### 2. Acronym detection
Most datasets have acronyms (e.g., National Education Longitudinal Study → NELS). So, we did acronym detection to detect dataset-names that are not included in the train-data labels. The following procedure was used to extract them.

    Make a list of words by splitting a text by space.
    If a word in the list is surrounded by () and has uppercase characters and no lowercase characters, it is detected as an acronym candidate.
    If the number of characters in the acronym candidate is less than the threshold, remove it.
    Extract a few words before the acronym candidate from the text as a dataset-name candidate.
    If the initial characters of each word in the dataset candidate can form the acronym candidate, detect them as a dataset-name/acronym pair. (The dataset candidate is allowed to have initial characters unrelated to the acronym candidate.)
    Extract only those dataset-names that contain keywords (study, studies, data, survey, panel, census, cohort, longitudinal, or registry).
    Exclude dataset-names that contain ban words (system, center, committee, etc.).
    Apply the clean_text function.
    Exclude dataset-name if Jaccard scores between the dataset-name and any train-data labels or acronym-detection labels are greater than or equal to 0.5.
    Perform string-matching to the train and test data with the detected dataset-names and count the number of occurrences among the texts of each dataset-name. Extract only those dataset-names whose count is above the threshold, because It is more likely to be a dataset-name if it appears in a lot of texts.
    Finally, perform string-matching using the extracted dataset-names. Only when a dataset name appears more than a threshold number of times in the text, it is detected as a label.

The acronym itself is also detected as a dataset-name. String matching is performed on the dataset-name and the acronym. The acronym is detected as a label only when it and its long name appear more than a threshold number of times in the text.

By this acronym detection, we get a score of 0.418 on the public LB and 0.436 on the private LB. Each threshold was chosen based on the public LB score.

#### 3. Acronym detection version 2
To obtain more dataset-names, we performed a more aggressive acronym detection. We extract words that contain uppercase characters and no lowercase characters from the texts as acronym candidates. We search a chunk of words that is valid as the full name of the acronym candidate among the entire text (the actual code is this). This acronym detection does not improve the leaderboard score because it detects many false-positive labels. But we used the dataset-names detected by it for the NER model's training, which we describe in the later step.

#### 4. String matching with dataset-names from external data
We used the external U.S. government’s dataset-names obtained from this notebook. To reduce false-positive, we apply some processing below.

    Ext Extract only those dataset-names that contain keywords (study, studies, etc.).
    Apply the clean_text function.
    Exclude dataset-name if Jaccard scores between the dataset-name and any train-data labels or acronym-detection labels are greater than or equal to 0.5.
    Exclude dataset-name if the number of words it contains is less than the threshold.
    Perform string-matching to the train and test data with the extracted dataset-names and count the number of occurrences among the texts of each dataset-name. Extract only those dataset-names whose count is above the threshold.
    Finally, perform string-matching using the extracted dataset-names.

This approach improved the score of the public LB score from 0.418 to 0.424 and the private LB score from 0.436 to 0.486.

#### 5. Dataset-name variation detection using named entity recognition (NER)
We attempted to train a NER model as a solution to this competition, using BERT or RoBERTa with the train data or the Rich Context competition data as training data. However, NER models never outperformed rule-based approaches. We think this is because a large number of true-positive labels are intentionally excluded from the provided train data. 

Therefore, the train data is incomplete as training data for machine learning. However, we found NER can be used to cover the weakness of string-matching. That is, NER is useful for detecting dataset-name variations that cannot be detected by string-matching. For example, National Education Longitudinal Study is sometimes quoted as National Educational Longitudinal Survey.

We used spacy library to train a NER model. We used the train-data label, the acronym-detection label, the acronym-detection-version-2 labels, and the external U.S. government label for training.

We detect dataset-name candidates from the test data using the trained NER model. We calculated Jaccard scores between dataset-name candidates and any train-data labels or acronym-detection labels. We selected candidates with Jaccard scores greater than or equal to 0.5 as variations of the existing labels. This approach improved the score of the public LB score from 0.424 to 0.440 and the private LB score from 0.486 to 0.504.

#### 6. String matching with dataset-names from the train data
   
Finally, we applied basic string-matching using the train-data labels. We also used the acronyms of the train-data labels for string-matching. This approach improved the score of the public LB score from 0.440 to 0.614 and the private LB score from 0.504 to 0.513.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: