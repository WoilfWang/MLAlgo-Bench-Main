You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Feedback_Prize_-_English_Language_Learning_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description

#### Goal of the Competition
The goal of this competition is to assess the language proficiency of 8th-12th grade English Language Learners (ELLs). Utilizing a dataset of essays written by ELLs will help to develop proficiency models that better supports all students.

Your work will help ELLs receive more accurate feedback on their language development and expedite the grading cycle for teachers. These outcomes could enable ELLs to receive more appropriate learning tasks that will help them improve their English language proficiency.

#### Context

Writing is a foundational skill. Sadly, it's one few students are able to hone, often because writing tasks are infrequently assigned in school. A rapidly growing student population, students learning English as a second language, known as English Language Learners (ELLs), are especially affected by the lack of practice. While automated feedback tools make it easier for teachers to assign more writing tasks, they are not designed with ELLs in mind. 

Existing tools are unable to provide feedback based on the language proficiency of the student, resulting in a final evaluation that may be skewed against the learner. Data science may be able to improve automated feedback tools to better support the unique needs of these learners.

Competition host Vanderbilt University is a private research university in Nashville, Tennessee. It offers 70 undergraduate majors and a full range of graduate and professional degrees across 10 schools and colleges, all on a beautiful campus—an accredited arboretum—complete with athletic facilities and state-of-the-art laboratories. Vanderbilt is optimized to inspire and nurture cross-disciplinary research that fosters discoveries that have global impact. Vanderbilt and co-host, The Learning Agency Lab, an independent nonprofit based in Arizona, are focused on developing science of learning-based tools and programs for social good.

Vanderbilt and The Learning Agency Lab have partnered together to offer data scientists the opportunity to support ELLs using data science skills in machine learning, natural language processing, and educational data analytics. You can improve automated feedback tools for ELLs by sensitizing them to language proficiency. The resulting tools could serve teachers by alleviating the grading burden and support ELLs by ensuring their work is evaluated within the context of their current language level.
                          

This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
Submissions are scored using MCRMSE, mean columnwise root mean squared error:
$$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$$
where Nt is the number of scored ground truth target columns, and y and ˆy are the actual and predicted values, respectively.

Submission File

For each text_id in the test set, you must predict a value for each of the six analytic measures (described on the Data page). The file should contain a header and have the following format:

    text_id,cohesion,syntax,vocabulary,phraseology,grammar,conventions
    0000C359D63E,3.0,3.0,3.0,3.0,3.0,3.0
    000BAD50D026,3.0,3.0,3.0,3.0,3.0,3.0
    00367BB2546B,3.0,3.0,3.0,3.0,3.0,3.0
    003969F4EDB6,3.0,3.0,3.0,3.0,3.0,3.0
    ...


##  Dataset Description:
Update
You may read more about Feedback Prize 3.0 data from the following publication:

Crossley, S. A., Tian, Y., Baffour, P., Franklin, A., Kim, Y., Morris, W., Benner, B., Picou, A., & Boser, U. (2023). Measuring second language proficiency using the English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus.  International Journal of Learner Corpus Research, 9 (2), 248-269. [link]


The dataset presented here (the ELLIPSE corpus) comprises argumentative essays written by 8th-12th grade English Language Learners (ELLs). The essays have been scored according to six analytic measures: cohesion, syntax,  vocabulary, phraseology, grammar, and conventions.
Each measure represents a component of proficiency in essay writing, with greater scores corresponding to greater proficiency in that measure. The scores range from 1.0 to 5.0 in increments of 0.5. Your task is to predict the score of each of the six measures for the essays given in the test set.
Some of these essays have appeared in the datasets for the Feedback Prize - Evaluating Student Writing and Feedback Prize - Predicting Effective Arguments competitions. You are welcome to make use of these earlier datasets in this competition.

File and Field Information

    train.csv - The training set, comprising the full_text of each essay, identified by a unique text_id. The essays are also given a score for each of the seven analytic measures above: cohesion, etc. These analytic measures comprise the target for the competition.
    test.csv - For the test data we give only the full_text of an essay together with its text_id.
    sample_submission.csv - A submission file in the correct format. See the Evaluation page for details.

Please note that this is a Code Competition. We give a few sample essays in test.csv to help you author your solutions. When your submission is scored, this example test data will be replaced with the full test set. The full test set comprises about 2700 essays.

train.csv - column name: text_id, full_text, cohesion, syntax, vocabulary, phraseology, grammar, conventions
test.csv - column name: text_id, full_text


## Dataset folder Location: 
../../kaggle-data/feedback-prize-english-language-learning. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
#### Overview
My solution is based on an ensemble of multiple finetuned NLP transformer models. Additionally, I employ two rounds of pseudo tagging on old Feedback data.

I follow a lot of our advice from the second Feedback competition described here and worked for around two weeks on this competition. I split my time working around 50% on accuracy, and 50% on efficiency solution.

#### Cross validation
In general, I observed very good correlation between local CV and public LB. As the data is very small and the metric is RMSE, the local scores can be quite shaky. To that end, for each experiment I was running, I trained three unique seeds and always only compared the average of these three seeds. So for example, if I would want to compare LR=1e-5 vs. LR=2e-5 I would run for each of those two experiments three separate seeds for a single fold, and only if the average of the three seeds improves, I would run on all my 5-folds, and then again compare 3-seed blends to make sure.

This allows to bring more trust to my experiments and as the data is really small, this was in general possible for me to do.

#### Modeling
The problem at hand is very much straight forward, feed in the text to a transformer model, apply some pooling, add a linear head, and predict regression targets. I used combinations of the following variations of the training routine for my final ensemble:

Token length:
512, 1024, 2048

All my models are trained and predicted with dynamic padding.
Pooling: CLS Token, GeM Pooling

Backbones:

    Deberta-V3-Base
    Deberta-V3-Large
    Deberta-V2-XL
    Deberta-V2-XXL
    Longformer Large
    Roberta-Large

I usually run 3 epochs for most of my models, all with cosine decay learning rate always picking the last epoch. I use differential learning rate for backbone and head of the model. I do not use any other techniques suggested in forums like differential lr across layers of backbone or reinitialization.

As always, I retrained my models on full data for final subs, but also blended some fold models in as I had lots of runtime left.

#### Ensembling
For most of my subs I just did usual average across seeds and models. My final best sub is a Nelder-Mead optimized ensemble of models, where I optimize the ensemble weights separately per target column. To not overfit too much on CV, I added weight bounds between 1 and 3 on the weights.

Actually, I could have trusted the local optimization even more, I have an unselected sub from couple of days ago with best local CV that would score #2 on private, which has unrestricted weights for the ensemble, also with negative weights, but it felt a bit too risky and as I only did one sub for best CV, I chose a bit more of a conservative one.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: