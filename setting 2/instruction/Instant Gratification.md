You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Instant_Gratification_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to Instant (well, almost) Gratification!
In 2015, Kaggle introduced Kernels as a resource to competition participants. It was a controversial decision to add a code-sharing tool to a competitive coding space. We thought it was important to make Kaggle more than a place where competitions are solved behind closed digital doors. Since then, Kernels has grown from its infancy--essentially a blinking cursor in a docker container--into its teenage years. We now have more compute, longer runtimes, better datasets, GPUs, and an improved interface.

We have iterated and tested several Kernels-only (KO) competition formats with a true holdout test set, in particular deploying them when we would have otherwise substituted a two-stage competition. However, the experience of submitting to a Kernels-only competition has typically been asynchronous and imperfect; participants wait many days after a competition has concluded for their selected Kernels to be rerun on the holdout test dataset, the leaderboard updated, and the winners announced. This flow causes heartbreak to participants whose Kernels fail on the unseen test set, leaving them with no way to correct tiny errors that spoil months of hard work.

Say Hello to Synchronous KO

We're now pleased to announce general support for a synchronous Kernels-only format. When you submit from a Kernel, Kaggle will run the code against both the public test set and private test set in real time. This small-but-substantial tweak improves the experience for participants, the host, and Kaggle:

    With a truly withheld test set, we are practicing proper, rigorous machine learning.
    We will be able to offer more varieties of competitions and intend to run many fewer confusing two-stage competitions.
    You will be able to see if your code runs successfully on the withheld test set and have the leeway to intervene if it fails.
    We will run all submissions against the private data, not just selected ones. Participants will get the complete and familiar public/private scores available in a traditional competition.
    The final leaderboard can be released at the end of the competition, without the delay of rerunning Kernels.

This competition is a low-stakes, trial-run introduction to our new synchronous KO implementation. We want to test that the process goes smoothly and gather feedback on your experiences. While it may feel like a normal KO competition, there are complicated new mechanics in play, such as the selection logic of Kernels that are still running when the deadline passes.

Since the competition also presents an authentic machine learning problem, it will also award Kaggle medals and points. Have fun, good luck, and welcome to the world of synchronous Kernels competitions!

##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each id in the test set, you must predict a probability for the target variable. The file should contain a header and have the following format:

    id,target
    ba88c155ba898fc8b5099893036ef205,0.5
    7cbab5cea99169139e7e6d8ff74ebb77,0.5
    7baaf361537fbd8a1aaa2c97a6d4ccc7,0.5
    etc.


##  Dataset Description:
This is an anonymized, binary classification dataset found on a USB stick that washed ashore in a bottle. There was no data dictionary with the dataset, but this poem was handwritten on an accompanying scrap of paper:

    Silly column names abound,  
    but the test set is a mystery.  
    Careful how you pick and slice,  
    or be left behind by history.  

Files

In a synchronous Kernels-only competition, the files you can observe and download will be different than the private test set and sample submission. The files may have different ids, may be a different size, and may vary in other ways, depending on the problem. You should structure your code so that it predicts on the public test.csv in the format specified by the public sample_submission.csv, but does not hard code aspects like the id or number of rows. When Kaggle runs your Kernel privately, it substitutes the private test set and sample submission in place of the public ones.

    train.csv - the training set. The column name: id, target, and other feature names. 
    test.csv - the test set (you must predict the target value for these variables). The column name: id, and other feature names. 
    sample_submission.csv - a sample submission file in the correct format

  

## Dataset folder Location: 
../../kaggle-data/instant-gratification. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
In the spirit of the competition I prepared this summary before the competition ended, so it is an instant (well, almost) solution summary. I hope my final placing will not be too embarrassing.

My final model kernel is here. It is a Gaussian Mixture model with several tweaks to accommodate for the make_classification() special structure. In general, GMM here works completely unsupervised on both train and test data, and the train set labels are used to assign each cluster to a 0/1 class.

First I modified M step in GMM to round means estimations to -1 or 1 values (just after calculating them), to align it with hypercube vertices of make_classification and assuming class-sep equals default value of 1. This change alone with 3 clusters per class gave me 0.974+.

The following 3 additional changes each gave me small improvements

    Running GMM with 2, 3 and 4 clusters per class and choosing the "best performing" one. This is a weak point of my model, I tried many approaches to make this identification better but it didn't work too well. At the end, I am not even completely sure that the number of clusters per class is a random number. My best model gives [46, 367, 99] distribution for 2 to 4 clusters per class respectively.

    Also on M step of GMM I re-weight probabilities of sample points in clusters to make total probabilities per cluster equal. This is motivated by the fact that make_classification() creates clusters with equal number of points.

    Increased covariance regularization in QMM from reg_covar=0.001 to 0.1

Now, I am really looking forward to see what other people have done. 80% of the work was published online, but this last 20% is still a meaningful challenge. Finally, thanks to Kaggle for giving us this beautiful puzzle :)



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: