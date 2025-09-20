You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named U.S._Patent_Phrase_to_Phrase_Matching_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Can you extract meaning from a large, text-based dataset derived from inventions? Here's your chance to do so.

The U.S. Patent and Trademark Office (USPTO) offers one of the largest repositories of scientific, technical, and commercial information in the world through its Open Data Portal. Patents are a form of intellectual property granted in exchange for the public disclosure of new and useful inventions. Because patents undergo an intensive vetting process prior to grant, and because the history of U.S. innovation spans over two centuries and 11 million patents, the U.S. patent archives stand as a rare combination of data volume, quality, and diversity.

“The USPTO serves an American innovation machine that never sleeps by granting patents, registering trademarks, and promoting intellectual property around the globe. The USPTO shares over 200 years' worth of human ingenuity with the world, from lightbulbs to quantum computers. Combined with creativity from the data science community, USPTO datasets carry unbounded potential to empower AI and ML models that will benefit the progress of science and society at large.”
— USPTO Chief Information Officer Jamie Holcombe

In this competition, you will train your models on a novel semantic similarity dataset to extract relevant information by matching key phrases in patent documents. Determining the semantic similarity between phrases is critically important during the patent search and examination process to determine if an invention has been described before. For example, if one invention claims "television set" and a prior publication describes "TV set", a model would ideally recognize these are the same and assist a patent attorney or examiner in retrieving relevant documents. This extends beyond paraphrase identification; if one invention claims a "strong material" and another uses "steel", that may also be a match. What counts as a "strong material" varies per domain (it may be steel in one domain and ripstop fabric in another, but you wouldn't want your parachute made of steel). We have included the Cooperative Patent Classification as the technical domain context as an additional feature to help you disambiguate these situations.

Can you build a model to match phrases in order to extract contextual information, thereby helping the patent community connect the dots between millions of patent documents?

This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
Submissions are evaluated on the Pearson correlation coefficient between the predicted and actual similarity scores.

Submission File

For each id (representing a pair of phrases) in the test set, you must predict the similarity score. The file should contain a header and have the following format:

    id,score
    4112d61851461f60,0
    09e418c93a776564,0.25
    36baf228038e314b,1
    etc.


##  Dataset Description:
In this dataset, you are presented pairs of phrases (an anchor and a target phrase) and asked to rate how similar they are on a scale from 0 (not at all similar) to 1 (identical in meaning). This challenge differs from a standard semantic similarity task in that similarity has been scored here within a patent's context, specifically its CPC classification (version 2021.05), which indicates the subject to which the patent relates. For example, while the phrases "bird" and "Cape Cod" may have low semantic similarity in normal language, the likeness of their meaning is much closer if considered in the context of "house".

This is a code competition, in which you will submit code that will be run against an unseen test set. The unseen test set contains approximately 12k pairs of phrases. A small public test set has been provided for testing purposes, but is not used in scoring.

Information on the meaning of CPC codes may be found on the USPTO website. The CPC version 2021.05 can be found on the CPC archive website.

Score meanings

The scores are in the 0-1 range with increments of 0.25 with the following meanings:

    1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
    0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
    0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
    0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
    0.0 - Unrelated.

Files

    train.csv - the training set, containing phrases, contexts, and their similarity scores
    test.csv - the test set set, identical in structure to the training set but without the score 
    sample_submission.csv - a sample submission file in the correct format

Columns

    id - a unique identifier for a pair of phrases
    anchor - the first phrase
    target - the second phrase
    context - the CPC classification (version 2021.05), which indicates the subject within which the similarity is to be scored
    score - the similarity. This is sourced from a combination of one or more manual expert ratings.


"Google Patent Phrase Similarity Dataset" by Google is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0)

train.csv - column name: id, anchor, target, context, score
test.csv - column name: id, anchor, target, context


## Dataset folder Location: 
../../kaggle-data/us-patent-phrase-to-phrase-matching. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
First of all, I would like to thank competition organizers for hosting this interesting competition. And thanks to my great teammate @Tifo , we discuss and work hard for the whole last month to explore new methods. And also thank to the community of great notebooks and discussions.

Where is magic

The key is that there exits strong correlations between different targets under the same anchor.  (you can see from the gap between groupkfold and kfold) For example, some targets are similar to the origin target and some are similar to the anchor. In short, adding them to the context can more effectively capture the correlation between the anchor and the target.

We used various methods to take advantage of this magic:

stage1

    Group the targets from the same anchor, such as 'target1, target2, target3, …'. Then add them to the context.
    Group the targets from the same anchor and context. This brings more relevant targets.
    Group the targets from the same anchor.  Group the anchors from the same context. Add them to the context in turn.

stage2

    Group the targets from the same anchor and add oof score to describe more specific quantitative information, like 'target1 23, target2 47, …'. The scores are multplied by 100 so can be recognized as a token.
    Group the targets from the same anchor and context, with score.

details

    During training, the group is performed inside the train-set, and the score is derived from the oof score from the first-stage models.
    During inference, the group is performed after concatenating train-set and test-set, and the score is derived from both the oof and the prediction of test-set from first-stage models. (Why concat? Because overlap anchors in train and test.)

Things that worked

FGM

    Adversarial-training in NLP 
    eps: 0.1
    single model cv 0.002-0.005

It's easy to add FGM to your training:
```python
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.1, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}

```
in training:
```python
 fgm = FGM(model)
 for batch_input, batch_label in data:
       loss = model(batch_input, batch_label)
       loss.backward()  

       # adversarial training
       fgm.attack() 
       loss_adv = model(batch_input, batch_label)
       loss_adv.backward() 
       fgm.restore()  

       optimizer.step()
       model.zero_grad()

```

EMA （Exponential Moving Average）

    decay: 0.999
    single model cv 0.001-0.003

Knowledge distillation

    In other words, soft label from ensemble oof. In this way, single model can achieve performance close to ensemble models (just save time but no more diversity)
    Make sure to use only the corresponding label for each fold to avoid leakage
    The actual performance of second or more rounds is almost the same as first round, and the cv will be distorted in a strange way. We only use few models distiled from the first round.

Models

    Deberta-v3-large
    Bert-for-patents
    Deberta-large

CV split
We use the 5fold StratifiedGroupKFold (the same seed 42, group by anchor).  So we are able to use OOF to get ensemble scores and model weights effectively. Linear regression is much faster than optuna search.
When there are enough models, our CV and LB are perfectly correlated. 

Notebook
submit: https://www.kaggle.com/code/zzy990106/upppm-final
You can find more details in the code..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: