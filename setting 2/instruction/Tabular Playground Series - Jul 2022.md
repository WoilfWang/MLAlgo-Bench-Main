You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Jul_2022_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to Kaggle's first ever unsupervised clustering challenge!

In this challenge, you are given a dataset where each row belongs to a particular cluster. Your job is to predict the cluster each row belongs to. You are not given any training data, and you are not told how many clusters are found in the ground truth labels. 

About the Tabular Playground Series

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.

The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model. These competitions are a great choice for people looking for something in between the Titanic Getting Started competition and the Featured competitions. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you; thus, we encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.


##  Evaluation Metric:
Submissions are evaluated on the Adjusted Rand Index between the ground truth cluster labels of the data and your predicted cluster labels. You are not given the number of ground truth clusters or any training labels. This is a completely unsupervised problem

Submission File

For each Id row in the data, you must predict the cluster of rows it belongs to Predicted. The file should contain a header and have the following format:

    Id,Predicted
    0,2
    1,1
    2,7
    etc.

##  Dataset Description:
For this challenge, you are given (simulated) manufacturing control data that can be clustered into different control states. Your task is to cluster the data into these control states. You are not given any training data, and you are not told how many possible control states there are. This is a completely unsupervised problem, one you might encounter in a real-world setting.
Good luck!

Files

    data.csv - the file includes continuous and categorical data; your task is to predict which rows should be clustered together in a control state
    sample_submission.csv - a sample submission file in the correct format, where Predicted is the predicted control state

data.csv - column name: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_27, f_28


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-jul-2022. In this folder, there are the following files you can use: sample_submission.csv, data.csv

## Solution Description:

Data structure.

Only 14 variables are relevant to clustering, the others are not used at all (this has been discovered early on in the competition). Variables f_07 - f_13 are integers (7 of them), and f_22 - f_28 are floating point (also 7 of them).

Floating point variables already appear to have a normal distribution, so there is no need to transform them.

But integer variables appear to come from some sort of mixed Poisson distribution. I tried to construct a model for them, but could not come up with anything that worked (that is, produced negative correlations but no negative values, and had a tractable probability mass function). So instead i did the same thing as everybody else - power-transformed them into a more 'normal' shape and treated them as multivariate normal vars. This is probably where the model could be improved the most [would be nice if competition organizers told us exactly how they constructed those vars, for future reference].

Cluster structure.

Data consists of 42 clusters, structured as 7 groups of 6 clusters each. Each cluster group is treated as one cluster as far as cluster labels are concerned, resulting in 7 distinct cluster labels.

Integer variables only vary by 7 cluster groups (so they truly form 7 clusters, not 42). Their means and covariances seem to be completely independent across 7 cluster groups (some covariances are zero).

Floating variables have two distinct distributions: for each cluster group, 2 floating variables have mean=0, standard deviation=1, and are uncorrelated with any other integer of floating point variables. Remaining 5 floating point variables have means that are -1 or +1, and full 5x5 covariance matrix. They are not correlated with integer variables, so full 14x14 covariance matrix consists of 7x7 block for integer vars, 5x5 block for correlated floating point vars, and 2x2 unit matrix for uncorrelated floating point vars.

Background

This data structure can be easily uncovered by running GaussianMixture(n_components=42) on floating point variables only, and looking at the means of mixture components.

I realized this when i looked at the source code of BayesianGMMClassifier (doesn't everybody do that? -:) ), trying to understand why it improved results so much, and saw that it used the same n_components=7 for each of the 7 clusters, resulting in total of 49 clusters. I thought that was a mistake, and made it use 1 component for each cluster, but that greatly reduced the LB score. That demonstrated that true data structure consists of > 7 clusters; further experimentation showed that each cluster group actually has 6 clusters, not 7.

Model.

I wrote a custom EM (Expectation Maximization) algorithm for a Gaussian Mixture Model, taking into account details of the data structure described above. I hard-coded means of all the floating point variables (0, -1, 1) to ensure convergence to true cluster centers. And this is what resulted in my best LB score..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: