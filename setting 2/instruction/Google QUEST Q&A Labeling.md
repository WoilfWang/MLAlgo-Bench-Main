You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Google_QUEST_Q&A_Labeling_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Computers are really good at answering questions with single, verifiable answers. But, humans are often still better at answering questions about opinions, recommendations, or personal experiences. 

Humans are better at addressing subjective questions that require a deeper, multidimensional understanding of context - something computers aren't trained to do well…yet.. Questions can take many forms - some have multi-sentence elaborations, others may be simple curiosity or a fully developed problem. They can have multiple intents, or seek advice and opinions. Some may be helpful and others interesting. Some are simple right or wrong. 

Unfortunately, it’s hard to build better subjective question-answering algorithms because of a lack of data and predictive models. That’s why the CrowdSource team at Google Research, a group dedicated to advancing NLP and other types of ML science via crowdsourcing, has collected data on a number of these quality scoring aspects.

In this competition, you’re challenged to use this new dataset to build predictive algorithms for different subjective aspects of question-answering. The question-answer pairs were gathered from nearly 70 different websites, in a "common-sense" fashion. Our raters received minimal guidance and training, and relied largely on their subjective interpretation of the prompts. As such, each prompt was crafted in the most intuitive fashion so that raters could simply use their common-sense to complete the task. By lessening our dependency on complicated and opaque rating guidelines, we hope to increase the re-use value of this data set. What you see is what you get!

Demonstrating these subjective labels can be predicted reliably can shine a new light on this research area. Results from this competition will inform the way future intelligent Q&A systems will get built, hopefully contributing to them becoming more human-like.

##  Evaluation Metric:
Submissions are evaluated on the mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

Submission File

For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

    qa_id,question_asker_intent_understanding,...,answer_well_written
    6,0.0,...,0.5
    8,0.5,...,0.1
    18,1.0,...,0.0
    etc.


##  Dataset Description:
The data for this competition includes questions and answers from various StackExchange properties. Your task is to predict target values of 30 labels for each question-answer pair.

The list of 30 target labels are the same as the column names in the sample_submission.csv file. Target labels with the prefix question_ relate to the question_title and/or question_body features in the data. Target labels with the prefix answer_ relate to the answer feature.

Each row contains a single question and a single answer to that question, along with additional features. The training data contains rows with some duplicated questions (but with different answers). The test data does not contain any duplicated questions.

This is not a binary prediction challenge. Target labels are aggregated from multiple raters, and can have continuous values in the range [0,1]. Therefore, predictions must also be in that range.

Since this is a synchronous re-run competition, you only have access to the Public test set. For planning purposes, the re-run test set is no larger than 10,000 rows, and less than 8 Mb uncompressed.

Additional information about the labels and collection method will be provided by the competition sponsor in the forum.

File descriptions

    train.csv - the training data (target labels are the last 30 columns)
    test.csv - the test set (you must predict 30 labels for each test set row)
    sample_submission.csv - a sample submission file in the correct format; column names are the 30 target labels

train.csv - column name: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence, question_type_definition, question_type_entity, question_type_instructions, question_type_procedure, question_type_reason_explanation, question_type_spelling, question_well_written, answer_helpful, answer_level_of_information, answer_plausible, answer_relevance, answer_satisfaction, answer_type_instructions, answer_type_procedure, answer_type_reason_explanation, answer_well_written
test.csv - column name: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host


## Dataset folder Location: 
../../kaggle-data/google-quest-challenge. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
#### models
1.Model structure。we design different models structure. We mainly refer to the solution of ccf internet sentiment analysis，concat different cls embedding . here it is the link BDCI2019-SENTIMENT-CLASSIFICATION

2.We found 30 labels through analysis, one is the question-related evaluation, and the other is the answer-related evaluation. In order to make the model learn better, we have designed the q model to remove the problem-related label and the a model to process the answer Related labels。 it is better than qa models.

3.different model test. roberta base >roberta large >xlnet base >bert base > t5 base.

#### Post-processing
Analysis and evaluation methods and competition data，we use 0,1 reseting way. it improve lb 0.05 or more.

#### Features
we want that our model learns features that are not only considered in text, so we add host and
category embeeding features annd other Statistical Features。 it improve both cv and lb about 0.005.

#### Text clean
We also did text cleaning to remove stop words and some symbols， it improve about 0.002

#### Stacking
Our Best Private model scored 0.42787 ,but we dont't select it. it is stacking by roberta large and roberta base and xlnet base.

blend.loc[:,targets] = roberta_large_oof_test.loc[:,targets].values*0.4+0.3*roberta_base_oof_test.loc[:,targets].values+\ xlnet_base_oof_test.loc[:,targets].values*0.3


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: