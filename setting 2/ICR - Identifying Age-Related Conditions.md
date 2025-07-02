You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named ICR_-_Identifying_Age-Related_Conditions_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
#### Goal of the Competition
The goal of this competition is to predict if a person has any of three medical conditions. You are being asked to predict if the person has one or more of any of the three medical conditions (Class 1), or none of the three medical conditions (Class 0). You will create a model trained on measurements of health characteristics.

To determine if someone has these medical conditions requires a long and intrusive process to collect information from patients. With predictive models, we can shorten this process and keep patient details private by collecting key characteristics relative to the conditions, then encoding these characteristics.

Your work will help researchers discover the relationship between measurements of certain characteristics and potential patient conditions.

#### Context
They say age is just a number but a whole host of health issues come with aging. From heart disease and dementia to hearing loss and arthritis, aging is a risk factor for numerous diseases and complications. The growing field of bioinformatics includes research into interventions that can help slow and reverse biological aging and prevent major age-related ailments. Data science could have a role to play in developing new methods to solve problems with diverse data, even if the number of samples is small.

Currently, models like XGBoost and random forest are used to predict medical conditions yet the models' performance is not good enough. Dealing with critical problems where lives are on the line, models need to make correct predictions reliably and consistently between different cases.

Founded in 2015, competition host InVitro Cell Research, LLC (ICR) is a privately funded company focused on regenerative and preventive personalized medicine. Their offices and labs in the greater New York City area offer state-of-the-art research space. InVitro Cell Research's Scientists are what set them apart, helping guide and defining their mission of researching how to repair aging people fast.

In this competition, you’ll work with measurements of health characteristic data to solve critical problems in bioinformatics. Based on minimal training, you’ll create a model to predict if a person has any of three medical conditions, with an aim to improve on existing methods.

You could help advance the growing field of bioinformatics and explore new methods to solve complex problems with diverse data.

This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
Submissions are evaluated using a balanced logarithmic loss. The overall effect is such that each class is roughly equally important for the final score.

Each observation is either of class 0 or of class 1. For each observation, you must submit a probability for each class. The formula is then:
$$\text{Log Loss} = \frac{-\frac{1}{N_{0}} \sum_{i=1}^{N_{0}} y_{0 i} \log  p_{0 i} - \frac {1}{N_{1}} \sum_{i=1}^{N_{1}} y_{1 i} \log  p_{1 i} } { 2 }$$
where (N_{c}) is the number of observations of class (c), (\log) is the natural logarithm, (y_{c i}) is 1 if observation (i) belongs to class (c) and 0 otherwise, (p_{c i}) is the predicted probability that observation (i) belongs to class (c).

The submitted probabilities for a given row are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum). In order to avoid the extremes of the log function, each predicted probability p is replaced with $\max(\min(p,1-10^{-15}),10^{-15})$.

Submission File

For each id in the test set, you must predict a probability for each of the two classes. The file should contain a header and have the following format:

    Id,class_0,class_1
    00eed32682bb,0.5,0.5
    010ebe33f668,0.5,0.5
    02fa521e1838,0.5,0.5
    040e15f562a2,0.5,0.5
    046e85c7cc7f,0.5,0.5
    ...


##  Dataset Description:
The competition data comprises over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions -- a binary classification problem.

Note that this is a Code Competition, in which the actual test set is hidden. In this version, we give some sample data in the correct format to help you author your solutions. When your submission is scored, this example test data will be replaced with the full test set. There are about 400 rows in the full test set.

Files and Field Descriptions

train.csv - The training set.

    Id Unique identifier for each observation.
    AB-GL Fifty-six anonymized health characteristics. All are numeric except for EJ, which is categorical.
    Class A binary target: 1 indicates the subject has been diagnosed with one of the three conditions, 0 indicates they have not.

test.csv - The test set. Your goal is to predict the probability that a subject in this set belongs to each of the two classes.

greeks.csv - Supplemental metadata, only available for the training set.

    Alpha Identifies the type of age-related condition, if present.
    A No age-related condition. Corresponds to class 0.
    B, D, G The three age-related conditions. Correspond to class 1.

Beta, Gamma, Delta Three experimental characteristics.

Epsilon The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected.

sample_submission.csv - A sample submission file in the correct format. See the Evaluation page for more details.

greeks.csv - column name: Id, Alpha, Beta, Gamma, Delta, Epsilon


## Dataset folder Location: 
../../kaggle-data/icr-identify-age-related-conditions. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv, greeks.csv

## Solution Description:
What we did:

    just like many people, we used time and max(time)+1 for test
    removed rows where time is None, I noticed a weird cluster that was far away from all other data when playing with umap and it were rows with absent time
    filling Nan values with -100, probably doesn't matter if its median or low numbers.
    used a technique I saw in some other competition: reducing dimensions with umap and then labeling clusters with kmeans, it didn't bring a lot of score, but it's there
    did feature permutation manually, dropped any cols that made score even slightly worse
    for a model I used Catboost, xgb with parameters from some public notebook and tabpfn. LGBM didn't seem to work for me as it always dropped my CV
    then we just averaged our predictions for test and that's it

Also, I want to mention that we wanted to try edit tabpfn to get embeddings and we had an idea to try fine tune tabpfn, but it didn't work out.

I also tried optuna for optimizing my models, but it didn't work out.

Stacking didn't help my score much as well


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: