You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tweet_Sentiment_Extraction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
"My ridiculous dog is amazing." [sentiment: positive]

With all of the tweets circulating every second it is hard to tell whether the sentiment behind a specific tweet will impact a company, or a person's, brand for being viral (positive), or devastate profit because it strikes a negative tone. Capturing sentiment in language is important in these times where decisions and reactions are created and updated in seconds.  But, which words actually lead to the sentiment description? In this competition you will need to pick out the part of the tweet (word or phrase) that reflects the sentiment.

Help build your skills in this important area with this broad dataset of tweets. Work on your technique to grab a top spot in this competition. What words in tweets support a positive, negative, or neutral sentiment? How can you help make that determination using machine learning tools?

In this competition we've extracted support phrases from Figure Eight's Data for Everyone platform. The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international licence. Your objective in this competition is to construct a model that can do the same - look at the labeled sentiment for a given tweet and figure out what word or phrase best supports it.

Disclaimer: The dataset for this competition contains text that may be considered profane, vulgar, or offensive.

##  Evaluation Metric:
The metric in this competition is the [word-level Jaccard score](https://en.wikipedia.org/wiki/Jaccard_index). A good description of Jaccard similarity for strings is [here](https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50).

A Python implementation based on the links above, and matched with the output of the C# implementation on the back end, is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric, then, is:
$$score = \frac{1}{n} \sum_{i=1}^n jaccard( gt\_i, dt\_i )$$
where:

    n=number of documents𝑛=number of documents
    jaccard=the function provided above
    gt_i=the ith ground truth
    dt_i=the ith prediction

## Submission File
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:

    textID,selected_text
    2,"very good"
    5,"I don't care"
    6,"bad"
    8,"it was, yes"
    etc.

##  Dataset Description:
What files do I need?

You'll need train.csv, test.csv, and sample_submission.csv.

What should I expect the data format to be?

Each row contains the text of a tweet and a sentiment label. In the training set you are provided with a word or phrase drawn from the tweet (selected_text) that encapsulates the provided sentiment.

Make sure, when parsing the CSV, to remove the beginning / ending quotes from the text field, to ensure that you don't include them in your training.

What am I predicting?

You're attempting to predict the word or phrase from the tweet that exemplifies the provided sentiment. The word or phrase should include all characters within that span (i.e. including commas, spaces, etc.). The format is as follows:

<id>,"<word or phrase that supports the sentiment>"

For example:

    2,"very good"
    5,"I am neutral about this"
    6,"bad"
    8,"if you say so!"
    etc.

Files

    train.csv - the training set
    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format

Columns

    textID - unique ID for each piece of text
    text - the text of the tweet
    sentiment - the general sentiment of the tweet
    selected_text - [train only] the text that supports the tweet's sentiment

train.csv - column name: textID, text, selected_text, sentiment
test.csv - column name: textID, text, sentiment


## Dataset folder Location: 
../../kaggle-data/tweet-sentiment-extraction. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
#### Main parts of the solution:

    • Pre-processing
    • Predicting 3 (start, end) candidates and assigning scores to them
    • Scoring them with external model and adding one more score to each candidate
    • Blending scores for candidates from different models
    • Selecting the best candidate
    • Post-processing

#### Pre-processing and post-processing.
“Magic” part. I came up with the algorithm, which is similar to those already described in great details in other solutions, so I will explain it only briefly.

Pre-processing is the procedure that reduces the amount of ‘noise’ in initial data and the goal of post-processing is to bring it back. They are based on the counting of extra spaces in the part of a tweet before selected text. Extra space is any space which is present in a tweet, but not in ‘ ’.join(tweet.split()).

So, in pre-processing I shift indices of selected text to the right on the amount of extra spaces and in post-processing to the left. The nuance here is that the length of selected text should be calculated with the single spaces around it. So in post-processing usually target end_idx is start_idx + len(selected_text)+2, but if selected_text ends with dot, comma, question mark etc. only space before the text should be taken into account and end_idx is start_idx + len(selected_text)+1.

#### Predicting 3 (start, end) candidates
Architecture

The model in this case is a transformer. I used BERT, RoBERTa and ELECTRA.

The input for the model is the following:

BERT or ELECTRA: [CLS] [POSITIVE] tweet [SEP]

RoBERTa: ``
‘[POSITIVE]’ can also be ‘[NEUTRAL]’ and ‘[NEGATIVE]’, these are added sentiment tokens.

Embeddings for them are initialized with the embeddings of corresponding words ‘positive’, ‘neutral’ and ‘negative’. At the early stage I also tried to put [SEP] between sentiment token and tweet, but it worked slightly worse. Did not experiment with this setup later.

As target each model gets indices of start and end tokens of selected text.

The model has four heads:
1) QA dense head (just a linear layer without any dropout) for predicting start and end tokens. Takes token representation as the concatenation of the corresponding hidden states from the last two layers of the transformer. Tried here to take weighted sum of hidden states from all layers with learnable weights, but it performed a bit worse.

Loss is computed with KL divergence to add label smoothing: true target token is given 0.9 probability and two of its neighbors (left and right) both take 0.05. If true target token is in the beginning of the sentence and we are calculating loss for start logits then true token still gets 0.9, but two following are taking 0.06 and 0.04. Analogous thing is implemented if true end token is the last: its proba is 0.9, but two previous have 0.06 and 0.04.

2) Linear layer to predict binary target for each token: if it should be in selected text or not. Takes hidden states from the last layer. Experimented with other layers a lot, but it did not improve the performance. The loss in binary cross-entropy.

3) Linear layer to predict a sentiment of each token. Also uses only the last layer of a transformer. Predicts 3 classes – neutral, positive and negative. Tokens from selected text are labeled as having the same sentiment as the tweet, while all other tokens are assigned neutral class. The loss here is the usual cross-entropy for each token separately.

4) Two linear layers with ReLU in between to predict the sentiment of the whole tweet. Concatenates mean and max pooling over all tokens in a tweet skipping cls and sentiment tokens. Then concatenates such representations from the last two layers of a transformer and passes it through the multi-sample dropout. Also utilizes the momentum exchange (arxiv 2002.11102) before calculating the loss with cross-entropy.

Training phase

During training, the total loss is calculated as the weighted sum of losses from all four heads. Training is performed on 8 folds with AdamW optimizer and using SWA over a get_cosine_with_hard_restarts_schedule_with_warmup scheduler for 10 epochs. SWA snapshots were taken at the end of each epoch, despite this steps did not coincide with the steps of the minimal learning rate for the combination of parameters I used (num_warmup_steps, num_cycles). And for some reason (maybe it increases diversity between snapshots?) it worked better than taking snapshots at the end of each learning rate cycle.

Tried to implement self-distillation from 2002.10345, which looks very interesting and promises to increase stability of the training, but it only made the performance quite significantly worse. Maybe did something wrong in implementation.

Inference phase

1) At the inference time, the first head is used to create a set of (start, end) candidates. First of all, each pair of (start, end) indices where end >= start is assigned a logit as a sum of individual start and end logits. All cases where end < start are given -999 logits. Then softmax is applied across all pairs to obtain probabilities for candidates and top 3 of them are selected to be used for the further processing. Tried other numbers of candidates, but 3 worked best. Let’s call the probability of a candidate from this head ‘qa_prob’.

2) The output of the second head is the set of logits: one for each token. To obtain a score for each of the selected (start, end) candidates I took the sigmoid from the tokens and calculated the average log of the resultant token probabilities across candidate tokens. Let’s call the output number as ‘score_per_token’.

3) The output of the third head is used in a very similar way to the previous. The only difference is that instead of sigmoid the softmax is taken over each token logits (there are 3 of them here – by the number of sentiments) and the proba corresponding to the sentiment of the tweet is selected. Then the same averaging operation as for previous head is applied to obtain a score for candidates. Let’s call it ‘sentiment_per_token’.
So in the end of this stage at inference time we have 3 (start, end) candidates with 3 scores assigned to each of them

#### External scorer
Architecture

Used ELECTRA with the following input:
[CLS] ([POSITIVE]|[NEUTRAL]|[NEGATIVE]) tweet [SEP] selected_text_candidate [SEP]

Single head (linear->tanh->dropout->linear) on top of the transformer is fed with the concatenation of the cls token hidden states from the last two layers to predict if the current candidate for selected text is correct or not. Loss is computed with cross-entropy after application of momentum exchange.

Tried to add a head for predicting a jaccard for a candidate along with or instead of classification head, but it made results worse.

Training phase

Dataset for training is built with all tweets each having three candidates from the previous model and also tweet with true selected_text is added if it is not present among candidates. Trained it for 3 epochs with AdamW and SWA.

Inference phase

3 candidates for each tweet are scored with this model. It ouputs two logits which are softmaxed and then the log of class 1 proba is taken as the score for the candidate. Will call it ‘external_score’ in the following.

So after this step we have 3 candidates and each of them has 4 scores.

#### Ensembling different transformers

BERT, RoBERTa and ELECTRA are actually ensembles of 8 (by the number of folds) models for which usual logits averaging is implemented. For BERT I used 4 bert-base-cased and 4 bert-large-cased models. For RoBERTa – 5 roberta-base-squad2 and 3 roberta-large. For ELECTRA – 6 electra-base-discriminator and 2 electra-large-discriminator.
External scorer is the combination of 4 electra-base-discriminator.

Each of three models – BERT, RoBERTa and ELECTRA – outputs 3 candidates for a given tweet. If there is an intersection between these sets of candidates, then only this intersection is considered. If intersection is empty then the union of BERT and ELECTRA candidates worked best.

The final score for each candidate is the weighted sum of qa_prob, score_per_token, sentiment_per_token and external_score inside the model type (BERT, RoBERTa or ELECTRA) and then the weighted (here models are weighted) sum of these sums. The final prediction is the candidate with the largest score, which then goes through post-processing. Also in case if there are two candidates with very close score, tried to predict their concatenation, but it did not really bring much.

The solution appeared to be quite sensitive to the weighting coefficients, which is disadvantage, because they cannot be reliably selected in cross-validation.

Eventual CV score for a wide range of weights was around 0.732. Individual performance among models was the best for ELECTRA. And RoBERTa scored better then BERT.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: