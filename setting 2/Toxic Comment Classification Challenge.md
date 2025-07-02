You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Toxic_Comment_Classification_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

The Conversation AI team, a research initiative founded by Jigsaw and Google (both a part of Alphabet) are working on tools to help improve online conversation. One area of focus is the study of negative online behaviors, like toxic comments (i.e. comments that are rude, disrespectful or otherwise likely to make someone leave a discussion). So far they’ve built a range of publicly available models served through the Perspective API, including toxicity. But the current models still make errors, and they don’t allow users to select which types of toxicity they’re interested in finding (e.g. some platforms may be fine with profanity, but not with other types of toxic content).

In this competition, you’re challenged to build a multi-headed model that’s capable of detecting different types of of toxicity like threats, obscenity, insults, and identity-based hate better than Perspective’s current models. You’ll be using a dataset of comments from Wikipedia’s talk page edits. Improvements to the current model will hopefully help online discussion become more productive and respectful.

Disclaimer: the dataset for this competition contains text that may be considered profane, vulgar, or offensive.

##  Evaluation Metric:

Submissions are now evaluated on the mean column-wise ROC AUC. In other words, the score is the average of the individual AUCs of each predicted column.

Submission File

For each id in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

    id,toxic,severe_toxic,obscene,threat,insult,identity_hate
    00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
    0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
    etc.

##  Dataset Description:
You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

    toxic
    severe_toxic
    obscene
    threat
    insult
    identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.

File descriptions

    train.csv - the training set, contains comments with their binary labels
    test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
    sample_submission.csv - a sample submission file in the correct format
    test_labels.csv - labels for the test data; value of -1 indicates it was not used for scoring; (Note: file added after competition close!)

Usage
The dataset under CC0, with the underlying comment text being governed by Wikipedia's CC-SA-3.0

train.csv - column name: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
test.csv - column name: id, comment_text


## Dataset folder Location: 
../../kaggle-data/jigsaw-toxic-comment-classification-challenge. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:

We’d like to begin by thanking Jigsaw/Kaggle for challenging us with a novel data science problem. And congratulations to everyone who made it through to the finish line - this was a marathon, not a sprint!

Summary of our approach:

    Diverse pre-trained embeddings                                     (baseline public LB of 0.9877)
    Translations as train/test-time augmentation (TTA)   (boosted LB from 0.9877 to 0.9880)
    Rough-bore pseudo-labelling (PL)                                  (boosted LB from 0.9880 to 0.9885)
    Robust CV + stacking framework                                   (boosted LB from 0.9885 to 0.9890)

We didn’t possess prior NLP-specific domain knowledge so our overall strategy was to test standard ML/DL techniques against the problem in a systematic fashion. We were gratified to see general techniques such as TTA and pseudo-labeling work effectively here. 

It would be remiss of us to not to mention that our hardware setup facilitated an exhaustive search of the solution space: we had 6 GPUs between the two of us. We had optimized code that allowed us to crunch 8-fold OOF data trained against 1+ million samples (TTA + PL) to convergence in approx. 2 hours per model. 

Our approach in detail
1. Diverse pre-trained embeddings 
   
Given that >90% of a model’s complexity resides in the embedding layer, we decided to focus on the embedding layer rather than the post-embedding layers. For the latter, our work-horse was two BiGru layers feeding into two final Dense layers. For the former, we searched the net for available pre-trained word embeddings and settled primarily on the highest-dimensional FastText and Glove embeddings pre-trained against Common Crawl, Wikipedia, and Twitter.

2. Translations as train/test-time augmentation (TTA)

We leveraged Pavel Ostyakov’s idea of machine translations to augment both train and test data sets using French, German, and Spanish translations translated back to English. Given the possibility of information leaks, we made sure to have translations stay on the same side of a train-val split as the original comment. For the predictions, we simply averaged the predicted probabilities of the 4 comments (EN, DE, FR, ES). 
This had a dramatic impact on the performance of our models. For example, 

    Vanilla Bi-GRU model:                 0.9862LB
    “ (w/ train-time augments):                  0.9867 LB
    “ (w/ test-time augments):                   0.9865 LB
    “ (w/ both train/test-time augments):   0.9874 LB

In other words, a single TTA-ed model was beating the majority of teams’ (presumably) ensembled submissions on the public ladder.

We were curious if this technique worked primarily by “fixing” non-English comments - we tested translating the original comments directly into English (which had the side-effect of translating non-English comments) and that resulted in lower performance than our full-form of augmentation.
3.  Rough-bore pseudo-labelling (PL)

We tried a number of PL variants - canonical per-batch updates, altering the loss functions etc. 

The variant that performed the best was simply labeling the test samples using our best-performing ensemble, adding them to the train set, and training to convergence. 

There’s been a fair amount of hay made on the forum about the difference in train and test distributions. PL helps with that.
4. Robust CV + stacking framework
For stacking, we used a weighted mean of arithmetic averaging and stacking, which worked marginally better (~.0001) than either approach alone. For stacking, we used primarily LightGBM, which both was faster than XGBoost and reached slightly better CV scores with heavy bayesian optimization. 

Parameters were selected by choosing the best out of 250 runs with bayesian optimization; key points in the parameters were small trees with low depth and strong l1 regularization. We bagged  6 runs of both DART and GBDT using different seeds to account for potential variance during stacking.

For CV, we tracked accuracy, log loss and AUC. A model was “strong” if when added to our stack it improved CV-log loss and CV-AUC in addition to improving the public board. We discarded a lot of models from our stack+blend for failing to do any of the three above in fear of overfitting. 

5. Miscellaneous takeaways/explorations
   
During our search of the solution space we tried a number of different approaches with varying success. A number key takeaways we thought would be helpful to share are below:

    Since most of the model complexity lay in the pre-trained embeddings, minor architecture changes made very little impact on score. Additional dense layers, gaussian vs. spatial dropout, additional dropout layers at the dense level, attention instead of max pooling, time distributed dense layers, and more barely changed the overall score of the model.

    Preprocessing was also not particularly impactful, although leaving punctuation in the embeddings for some models (with fasttext, so they could be accomodated) was helpful in stacking.

    Many comments were toxic only in the last sentences -- adding some models trained with the ending 25-50 characters in addition to the starting 200-300 assisted our stack.  

    Some approaches struggled to deal with the “ordering” problem of words. The same phrase with two words swapped can mean completely different things. This meant that CNN approaches were difficult to work with, as they rely on max-pooling as a crutch. Our best CNN (a wavenet-like encoder connected to some time distributed dense and dense layers) scored about .0015 lower than our best RNN. 

    Overall, other architectures struggled to achieve comparable performance to RNN. The only one that reached RNN levels was Attention Is All You Need, and it took significantly longer to train. 

    On that note, despite all the hate, tensorflow is superior to Keras in a lot of ways-- try implementing Attention Is All You Need or scalable models in Keras for a practical demonstration of that (yes, Keras just got tensorflow-served). That said, Keras is also fantastic, especially for rapid iteration, and what we used for our primary RNN models. 

    Kagglers often don’t want to mix models with different OOF splits, and it’s hard to understand why. It can make for over-optimistic CV predictions, but you won’t overfit the public board compared to the private with that strategy. As long as you are looking at directionality and not comparing stacks from different OOF splits to stacks from a single OOF split you shouldn’t have any issues. 

Cheers! Expect to see more of the Toxic Crusaders in future competitions!
P.S. I'm on the lookout for interesting Data Science/ML jobs so if you're privy to one please DM me!.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: