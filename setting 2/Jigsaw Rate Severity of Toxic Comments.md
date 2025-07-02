You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Jigsaw_Rate_Severity_of_Toxic_Comments_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
In Jigsaw's fourth Kaggle competition, we return to the Wikipedia Talk page comments featured in our first Kaggle competition. When we ask human judges to look at individual comments, without any context, to decide which ones are toxic and which ones are innocuous, it is rarely an easy task. In addition, each individual may have their own bar for toxicity. We've tried to work around this by aggregating the decisions with a majority vote. But many researchers have rightly pointed out that this discards meaningful information.

😄 🙂 😐 😕 😞
A much easier task is to ask individuals which of two comments they find more toxic. But if both comments are non-toxic, people will often select randomly. When one comment is obviously the correct choice, the inter-annotator agreement results are much higher.

In this competition, we will be asking you to score a set of about fourteen thousand comments. Pairs of comments were presented to expert raters, who marked one of two comments more harmful — each according to their own notion of toxicity. In this contest, when you provide scores for comments, they will be compared with several hundred thousand rankings. Your average agreement with the raters will determine your individual score. In this way, we hope to focus on ranking the severity of comment toxicity from innocuous to outrageous, where the middle matters as much as the extremes.

Can you build a model that produces scores that rank each pair of comments the same way as our professional raters?

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive. 

Related Work

The paper "Ruddit: Norms of Offensiveness for English Reddit Comments" by Hada et al. introduced a similar dataset that involved tuples of four sentences that were marked with best-worst scoring, and this data may be directly useful for building models.

We also note "Constructing Interval Variables via Faceted Rasch Measurement
and Multitask Deep Learning: a Hate Speech Application" by Kennedy et al. which compares a variety of different rating schemes and argues that binary classification as typically done in NLP tasks discards valuable information. Combining data from multiple sources, even with different annotation guidelines, may be essential for success in this competition.

Resources

The English language resources from our first Kaggle competition, and our second Kaggle competition, which are both available in the TensorFlow datasets Wikipedia Toxicity Subtypes and Civil Comments can be used to build models.
One example of a starting point is the open source UnitaryAI model.

Google Jigsaw

Google's Jigsaw team explores threats to open societies and builds technology that inspires scalable solutions. One Jigsaw product is PerspectiveAPI which is used by publishers and platforms worldwide as part of their overall moderation strategy.


This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
Submissions are evaluated on Average Agreement with Annotators. For the ground truth, annotators were shown two comments and asked to identify which of the two was more toxic. Pairs of comments can be, and often are, rated by more than one annotator, and may have been ordered differently by different annotators.

For each of the approximately 200,000 pair ratings in the ground truth test data, we use your predicted toxicity score to rank the comment pair. The pair receives a 1 if this ranking matches the annotator ranking, or 0 if it does not match.

The final score is the average across all the pair evaluations.

Please note the following:

    score is not constrained to any numeric range (e.g., you can predict [0, 1] or [-999, 999]).
    There is no tie breaking; tied comment scores will always be evaluated as 0. You could consider using something like scipy.stats.rankdata to force unique value.

Submission File

For each comment_id found in the comments_to_score.csv file, you must predict the toxic severity score associated with the comment text. The submission file should contain a header and have the following format:

    comment_id,score
    114890,0.43
    732895,0.98
    1139051,0.27
    etc.


##  Dataset Description:
In this competition you will be ranking comments in order of severity of toxicity. You are given a list of comments, and each comment should be scored according to their relative toxicity. Comments with a higher degree of toxicity should receive a higher numerical value compared to comments with a lower degree of toxicity.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive. 

Note, there is no training data for this competition. You can refer to previous Jigsaw competitions for data that might be useful to train models. But note that the task of previous competitions has been to predict the probability that a comment was toxic, rather than the degree or severity of a comment's toxicity.

    Toxic Comment Classification Challenge
    Jigsaw Unintended Bias in Toxicity Classification
    Jigsaw Multilingual Toxic Comment Classification

While we don't include training data, we do provide a set of paired toxicity rankings that can be used to validate models.

Files

    comments_to_score.csv - for each comment text in this file, your task is to predict a score that represents the relative toxic severity of the comment. Comments with a higher degree of toxicity should receive a higher numerical value compared to comments with a lower degree of toxicity; scores are relative, and not constrained to a certain range of values. NOTE: the rerun version of this file has ~14k comments that will be scored by your submitted model.
    sample_submission.csv - a sample submission file in the correct format
    validation_data.csv - pair rankings that can be used to validate models; this data includes the annotator worker id, and how that annotator ranked a given pair of comments; note, this data contains comments that are not found in comments_to_score.


validation_data.csv - column name: worker, less_toxic, more_toxic
comments_to_score.csv - column name: comment_id, text


## Dataset folder Location: 
../../kaggle-data/jigsaw-toxic-severity-rating. In this folder, there are the following files you can use: validation_data.csv, comments_to_score.csv, sample_submission.csv

## Solution Description:
Toxic Solution and Review (2nd Place)
First of all, congrats to the winners and everyone who enjoyed the competition! ! Although this is my first time participating in a competition, I really like this competition on Kaggle, especially the competitive atomsphere in there.

### Overview
To be honest I never thought of finishing in 2nd place, which is largely due to the luck, but I will say ''Thank you!'' to the luck for seeing positive results for my persistence with ''Bert'' Method.I have to say that we could marvel at and believe the performance of "Bert" in NLP world all the time. The **Roberta **helps me attain the success, even though its' public score is lower than linear model, but I find that the pinpoint of this competition perhaps is that **''Public leaderboard perhaps mislead us to be overfitting'' **

### Retrospective
What I remember fresh is probably in the first ten days of January, when many people started to increase the public leaderboard ranking crazily, which made me feel that the overall trend was not normal, so I made two decisions:

    Running program Locally in order to avoid paying more attention to the public leaderboard scores.
    I chose to part ways with a partner who was supposed to be teaming up. (The story here is that we communicated and did a lot of groundwork and exploration togethera at the beginning, but in those crazy period in January, I disagree with his obsession with improving public scoring and trust linear model instead of BERT, (Perharps the submission score of Roberta only 0.79416, which is even lower than 0.8 and not to say the linear model(0.893+)), but it would cause some overfitting risks I realized when I learn more from papers and my professor.
    Obviously, that's the reason why we can see this remarkable 'Shake-up' from Public leaderboard to Private Leaderboard. Fortunately, the Luck resonate my persistence in BERT method.

### About Dataset
I used data from the last toxic classification competition. Since the goal of this competition is to predict the toxicity of comments, which is a regression task, I weighted averaged several features of the orginal data.

### About Data Cleaning
BERT uses raw data when pretrained, which means that Pre-trained BERT is based on an uncleaned corpus. Actually, I find that Data Cleaning will change distribution and lose information.

Usually, data cleaning can improve the performance of linear models, but could harm the performance of BERT. If the weight decay or dropout value is too large, the model performance will degrade. When we go deep, we could find that Linear models need to be cleaned to reduce feature size and reduce overfitting. The capacity of BERT is large enough that it is not very necessary.

So in the end, I make a huge change of the code, especially simpliying the data cleaning procedure

Comparison：
Apply Data cleaning： Private Score：0.80826， Public Score：0.77609
Delete some Data cleaning： Private Score：0.81363， Public Score：0.79416

### About Feature Engineering
1.For text preprocessing, only simple tokenization methods are used. I didn't select the features manually because the model can learn the importance of each token.
2.The BERT model acts like a feature transformer, converting text data into a vector representation of length 768.

### About Train Method
I used BERT(Bidirectional Encoder Representations from Transformers) for this competition. BERT is currently the State-of-the-art language model that makes use of Transformer to learn contextual relations between words (or sub-words) in a text. The model I used in this competition consists of two parts: the RoBERTa base, and a multi-layer perceptron built upon it for regression. That‘s is the essence of idea why I choose this method and insist it until the competition is ended.

### About Esemble Weight
I randomly sampled 5 folds of data, each fold contained a proportion of the data and trained a separate Model on each fold. Then average up the predictions from the 5 models.

### Optimization
I used **Adam optimizer **as it is computationally efficient and works well with large data sets and large parameters. Besides, I also used a learning rate schedular to decrease the learning rate gradually based on the epoch. This has the effect of quickly learning good weights early and fine tuning them later.

### Avoid Overfitting
Focusing on using** Weight Decay, Dropout Layers and Early stopping **to prevent the models from overfitting. All of these keep the relative good generalization of model.
To be more specific, For *Earlystoping*, the stopping criteria will be based on the validation loss in each epoch.

Thanks everyone，I am happy everyone help new bird , I learn a lot from this competition!

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: