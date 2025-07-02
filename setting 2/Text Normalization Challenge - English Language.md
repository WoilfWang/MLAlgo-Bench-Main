You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Text_Normalization_Challenge_-_English_Language_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
As many of us can attest, learning another language is tough. Picking up on nuances like slang, dates and times, and local expressions, can often be a distinguishing factor between proficiency and fluency. This challenge is even more difficult for a machine.

Many speech and language applications, including text-to-speech synthesis (TTS) and automatic speech recognition (ASR), require text to be converted from written expressions into appropriate "spoken" forms. This is a process known as text normalization, and helps convert 12:47 to "twelve forty-seven" and $3.16 into "three dollars, sixteen cents." 

However, one of the biggest challenges when developing a TTS or ASR system for a new language is to develop and test the grammar for all these rules, a task that requires quite a bit of linguistic sophistication and native speaker intuition.

    A    <self>
    baby    <self>
    giraffe    <self>
    is    <self>
    6ft    six feet
    tall    <self>
    and    <self>
    weighs    <self>
    150lb    one hundred fifty pounds
    .    sil

In this competition, you are challenged to automate the process of developing text normalization grammars via machine learning. This track will focus on English, while a separate will focus on Russian here: Russian Text Normalization Challenge

##  Evaluation Metric:
Submissions are evaluated on prediction accuracy (the total percent of correct tokens). The predicted and actual string must match exactly in order to count as correct.  In other words, we are measuring sequence accuracy, in that any error in the output for a given token in the input sequence means that that error is wrong. For example, if the input is "145" and the predicted output is "one forty five" but the correct output is "one hundred forty five", this is counted as a single error. 

Submission File

For each token (id) in the test set, you must predict the normalized text. The file should contain a header and have the following format:

    id,after
    0_0,"the"
    0_1,"quick
    "0_2,"fox"
    ...

##  Dataset Description:
You are provided with a large corpus of text. Each sentence has a sentence_id. Each token within a sentence has a token_id. The before column contains the raw text, the after column contains the normalized text. The aim of the competition is to predict the after column for the test set. The training set contains an additional column, class, which is provided to show the token type. This column is intentionally omitted from the test set. In addition, there is an id column used in the submission format. This is formed by concatenating the sentence_id and token_id with an underscore (e.g. 123_5).

File descriptions

    en_sample_submission.csv - a submission file showing the correct format
    en_test.csv - the test set, does not contain the normalized text
    en_train.csv - the training set, contains the normalized text

en_test.csv - column name: sentence_id, token_id, before
en_train.csv - column name: sentence_id, token_id, class, before, after


## Dataset folder Location: 
../../kaggle-data/text-normalization-challenge-english-language. In this folder, there are the following files you can use: en_test.csv, en_train.csv, en_sample_submission.csv

## Solution Description:
First, congratulations to the top 3 teams!

Finishing 4th probably is not the best thing what could happen. But… not the worst either. And competition itself was interesting, not a typical machine learning competition. Thanks to organizers!

I started to look at this competition only week before the end, so I didn't have much time to build something big and complicated. 

My solution is very simple and basically consists of 3 text processing layers - each with different approach and tasks.

1. layer - statistical approach. I collected statistics from training data about possible transformations for each word, taking into account also it's preceding and following words. And I calculated "confidence" level for each such transformation (depending on how much other alternatives were possible, what were counts of examples for each alternative, etc).

If statistical model was "confident enough" about needed transformation I used it. This layer mostly handled plain texts and common transformations like "dr" to "doctor" etc

2. layer - pattern based approach. Or, simply speaking, regular expressions.
   
This layer handled data with known format - dates, times, numbers, phones, URLs.

3. layer - ML approach. I used several LightGBM models for deciding on ambiguous cases, where 1st layer was not confident enough about solution and also 2nd layer couldn't help.

This layer mostly made binary decisions between alternatives from 1.st layer, like whether "x" transforms to "x" or "by" and whether "-" is "-" or "to", etc.

Lack of time was the biggest problem for me - now I see that I've missed some trivial cases (which could be captured better if I had spent more time to analyze data). Like that 4 digits could actually be a number not a year and so on.

For time economy I also used too simple local validation and took some decisions based on public LB (which was only 1% of data!). But - real programmers test in production and real data scientists validate on test data ;).


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: