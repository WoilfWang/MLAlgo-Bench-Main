You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Bag_of_Words_Meets_Bags_of_Popcorn_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
In this tutorial competition, we dig a little "deeper" into sentiment analysis. Google's Word2Vec is a deep-learning inspired method that focuses on the meaning of words. Word2Vec attempts to understand meaning and semantic relationships among words. It works in a way that is similar to deep approaches, such as recurrent neural nets or deep neural nets, but is computationally more efficient. This tutorial focuses on Word2Vec for sentiment analysis.

Sentiment analysis is a challenging subject in machine learning. People express their emotions in language that is often obscured by sarcasm, ambiguity, and plays on words, all of which could be very misleading for both humans and computers. There's another Kaggle competition for movie review sentiment analysis. In this tutorial we explore how Word2Vec can be applied to a similar problem.
Deep learning has been in the news a lot over the past few years, even making it to the front page of the New York Times. These machine learning techniques, inspired by the architecture of the human brain and made possible by recent advances in computing power, have been making waves via breakthrough results in image recognition, speech processing, and natural language tasks. Recently, deep learning approaches won several Kaggle competitions, including a drug discovery task, and cat and dog image recognition.

**Tutorial Overview**
This tutorial will help you get started with Word2Vec for natural language processing. It has two goals: 

Basic Natural Language Processing: Part 1 of this tutorial is intended for beginners and covers basic natural language processing techniques, which are needed for later parts of the tutorial.

Deep Learning for Text Understanding: In Parts 2 and 3, we delve into how to train a model using Word2Vec and how to use the resulting word vectors for sentiment analysis.

Since deep learning is a rapidly evolving field, large amounts of the work has not yet been published, or exists only as academic papers. Part 3 of the tutorial is more exploratory than prescriptive -- we experiment with several ways of using Word2Vec rather than giving you a recipe for using the output.

To achieve these goals, we rely on an IMDB sentiment analysis data set, which has 100,000 multi-paragraph movie reviews, both positive and negative. 

**Acknowledgements**
This dataset was collected in association with the following publication:
    Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). "Learning Word Vectors for Sentiment Analysis." The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011). (link)

Please email the author of that paper if you use the data for any research applications. The tutorial was developed by Angela Chapman during her summer 2014 internship at Kaggle.

##  Evaluation Metric:
Metric: Submissions are judged on area under the ROC curve. 

**Submission Instructions**

You should submit a comma-separated file with 25,000 row plus a header row. There should be 2 columns: "id" and "sentiment", which contain your binary predictions: 1 for positive reviews, 0 for negative reviews. For an example, see "sampleSubmission.csv" on the Data page. 
    
    id,sentiment
    123_45,0 
    678_90,1
    12_34,0
    ...

##  Dataset Description:
**Data Set**

The labeled data set consists of 50,000 IMDB movie reviews, specially selected for sentiment analysis. The sentiment of reviews is binary, meaning the IMDB rating < 5 results in a sentiment score of 0, and rating >=7 have a sentiment score of 1. No individual movie has more than 30 reviews. The 25,000 review labeled training set does not include any of the same movies as the 25,000 review test set. In addition, there are another 50,000 IMDB reviews provided without any rating labels.

**File descriptions**

    labeledTrainData - The labeled training set. The file is tab-delimited and has a header row followed by 25,000 rows containing an id, sentiment, and text for each review.  
    testData - The test set. The tab-delimited file has a header row followed by 25,000 rows containing an id and text for each review. Your task is to predict the sentiment for each one. 
    unlabeledTrainData - An extra training set with no labels. The tab-delimited file has a header row followed by 50,000 rows containing an id and text for each review. 
    sampleSubmission - A comma-delimited sample submission file in the correct format.

**Data fields**

    id - Unique ID of each review
    sentiment - Sentiment of the review; 1 for positive reviews and 0 for negative reviews
    review - Text of the review

Code: Full tutorial code lives in this github repo.

testData.csv - column name: id, review
unlabeledTrainData.csv - column name: id, review
labeledTrainData.csv - column name: id, sentiment, review


## Dataset folder Location: 
../../kaggle-data/word2vec-nlp-tutorial. In this folder, there are the following files you can use: sampleSubmission.csv, testData.csv, unlabeledTrainData.csv, labeledTrainData.csv

## Solution Description:
The solution of **vgng** team, who ranked at 3rd position in the private leaderboard with the score of 0.97663. The solution is based on the paper:

    Mesnil, Grégoire, Tomas Mikolov, Marc'Aurelio Ranzato, and Yoshua Bengio. "Ensemble of Generative and Discriminative Techniques for Sentiment Analysis of Movie Reviews." ICLR 2015

This paper combines both generative and discriminative models for sentiment prediction.

### Generative Model
A generative model defines a distribution over the input. By training a generative model for each class, we can then use Bayes rule to predict which class a test sample belongs to. More formally, given a dataset of pairs ${x(i), y(i)}, i=1 ,..., N$ where $x(i)$ is the i-th document in the training set, $y(i) \in \{−1, +1\}$ is the corresponding label and N is the number of training samples, we train two models: $p^+(x|y = +1)$ for {$x(i)$ subject to $y(i) = +1$} and $p^−(x|y = −1)$ for {x subject to y= −1}. Then, given an input x at test time we compute the ratio (derived from Bayes rule): $r = p^+(x|y =+1)/p^−(x|y = −1) \times p(y = +1)/p(y = −1)$. If $r > 1$, then x is assigned to the positive class, otherwise to the negative class.

We have a few different choices of distribution we can choose from. The most common one is the n-gram, a count-based non-parametric method to compute $p(x^{(i)}_k|x^{(i)}_{k−1}, x^{(i)}_{k−2}, . . . , x^{(i)}_{k−N+1})$, where $x^{(i)}_k$ is the k-th word in the i-th document. In order to compute the likelihood of a document, we use the Markov assumption and simply multiply the n-gram probabilities over all words in the document: $p(x(i)) = \prod^K_{k=1} p(x^{(i)}_k|x^{(i)}_{k−1}, x^{(i)}_{k−2}, . . . , x^{(i)}_{k−N+1})$. As mentioned before, we train one n-gram language model using the positive documents and one model using the negative ones.

In our experiments, we used SRILM toolkit (Stolcke et al., 2002) to train the n-gram language models using modified Kneser-Ney smoothing (Kneser & Ney, 1995). Furthermore, as both language models are trained on different datasets, there is a mismatch between vocabularies: some words can appear only in one of the training sets. This can be a problem during scoring, as the test data contain novel words that were not seen in at least one of the training datasets. To avoid this problem, it is needed to add penalty during scoring for each out of vocabulary word.

N-grams are a very simple data-driven way to build language models. However, they suffer from both data sparsity and large memory requirement. Since the number of word combinations grows exponentially with the length of the context, there is always little data to accurately estimate probabilities for higher order n-grams.

In contrast with N-grams languages models, Recurrent neural networks (RNNs) (Mikolov et al., 2010) are parametric models that can address these issues. The inner architecture of the RNNs gives them potentially infinite context window, allowing them to perform smoother predictions.

We know that in practice, the context window is limited due to exploding and vanishing gradients (Pascanu et al., 2012). Still, RNNs outperform significantly n-grams and are the state of the art for statistical language modeling. A review of these techniques is beyond the scope of this short paper and we point the reader to (Mikolov, 2012) for a more in depth discussion on this topic.

Both when using n-grams and RNNs, we compute the probability of the test document belonging to the positive and negative class via Bayes’ rule. These scores are then averaged in the ensemble with other models, as explained in Section 2.4.

### LINEAR CLASSIFICATION OF WEIGHTED N-GRAM FEATURES
Among purely discriminative methods, the most popular choice is a linear classifier on top of a bagof-word representation of the document. 
The input representation is usually a tf-idf weighted word counts of the document. In order to preserve local ordering of the words, a better representation would consider also the position-independent n-gram counts of the document (bag-of-n-grams).

In our ensemble, we used a supervised reweighing of the counts as in the Naive Bayes Support Vector Machine (NB-SVM) approach (Wang & Manning, 2012). This approach computes a log-ratio vector between the average word counts extracted from positive documents and the average word counts extracted from negative documents. The input to the logistic regression classifier corresponds to the log-ratio vector multiplied by the binary pattern for each word in the document vector. Note that the logictic regression can be replaced by a linear SVM. 

Our implementation slightly improved the performance reported in (Wang & Manning, 2012) by adding tri-grams (improvement of +0.6%).

### SENTENCE VECTORS
Recently, (Le & Mikolov, 2014) proposed an unsupervised method to learn distributed representations of words and paragraphs. The key idea is to learn a compact representation of a word or paragraph by predicting nearby words in a fixed context window. This captures co-occurence statistics and it learns embeddings of words and paragraphs that capture rich semantics. Synonym words and similar paragraphs often are surrounded by similar context, and therefore, they will be mapped into nearby feature vectors (and vice versa).

Such embeddings can then be used to represent a new document (for instance, by averaging the representations of the paragraphs that constitute the document) via a fixed size feature vector. The authors then use such a document descriptor as input to a one hidden layer neural network for sentiment discrimination.

### MODEL ENSEMBLE
In this work, we combine the log probability scores of the above mentioned models via linear interpolation. More formally, we define the overall probability score as the weighted geometric mean of baseline models: $p(y = +1|x) = \prod p^k(y = +1|x)^{α_k}$, with $α_k > 0$.

We find the best setting of weights via brute force grid search, quantizing the coefficient values in the interval [0, 1] at increments of 0.1. The search is evaluated on a validation set to avoid overfitting.

We do not focus on a smarter way to find the α since we consider only 3 models in our approach and we consider it out of the scope of this paper. Using more models would make the use of such method prohibitive. For a larger number of models, one might want to consider random search of the α coefficients or even Bayesian approaches as these techniques will give better running time
performance.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: