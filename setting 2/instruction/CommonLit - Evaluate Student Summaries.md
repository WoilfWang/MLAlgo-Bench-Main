You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named CommonLit_-_Evaluate_Student_Summaries_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description

#### Goal of the Competition

The goal of this competition is to assess the quality of summaries written by students in grades 3-12. You'll build a model that evaluates how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary. You'll have access to a collection of real student summaries to train your model.

Your work will assist teachers in evaluating the quality of student work and also help learning platforms provide immediate feedback to students.

#### Context
Summary writing is an important skill for learners of all ages. Summarization enhances reading comprehension, particularly among second language learners and students with learning disabilities. Summary writing also promotes critical thinking, and it’s one of the most effective ways to improve writing abilities. However, students rarely have enough opportunities to practice this skill, as evaluating and providing feedback on summaries can be a time-intensive process for teachers. Innovative technology like large language models (LLMs) could help change this, as teachers could employ these solutions to assess summaries quickly.

There have been advancements in the automated evaluation of student writing, including automated scoring for argumentative or narrative writing. However, these existing techniques don't translate well to summary writing. Evaluating summaries introduces an added layer of complexity, where models must consider both the student writing and a single, longer source text. Although there are a handful of current techniques for summary evaluation, these models have often focused on assessing automatically-generated summaries rather than real student writing, as there has historically been a lack of these types of datasets.

Competition host CommonLit is a nonprofit education technology organization. CommonLit is dedicated to ensuring that all students, especially students in Title I schools, graduate with the reading, writing, communication, and problem-solving skills they need to be successful in college and beyond. The Learning Agency Lab, Vanderbilt University, and Georgia State University join CommonLit in this mission.

As a result of your help to develop summary scoring algorithms, teachers and students alike will gain a valuable tool that promotes this fundamental skill. Students will have more opportunities to practice summarization, while simultaneously improving their reading comprehension, critical thinking, and writing abilities.  


##  Evaluation Metric:
Submissions are scored using MCRMSE, mean columnwise root mean squared error:
$$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\left(\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \widehat{y}_{ij})^2\right)^{1/2}$$
where Nt is the number of scored ground truth target columns, and y and ˆy are the actual and predicted values, respectively.

Submission File

For each student_id in the test set, you must predict a value for each of the two analytic measures (described on the Data page). The file should contain a header and have the following format:

    student_id,content,wording
    000000ffffff,0.0,0.0
    111111eeeeee,0.0,0.0
    222222cccccc,0.0,0.0
    333333dddddd,0.0,0.0
    ...


##  Dataset Description:
Update
You may read more about Commonlit Summary Data from the following publication:

    Crossley, S., Baffour, P., Dascalu, M., & Ruseti, S. (2024). A World CLASSE Student Summary Corpus.  In Proceedings of 19th Workshop on Innovative Use of NLP for Building Educational Applications (BEA). [link]


The dataset comprises about 24,000 summaries written by students in grades 3-12 of passages on a variety of topics and genres. These summaries have been assigned scores for both content and wording. The goal of the competition is to predict content and wording scores for summaries on unseen topics.

File and Field Information

summaries_train.csv - Summaries in the training set.

    student_id - The ID of the student writer.
    prompt_id - The ID of the prompt which links to the prompt file.
    text - The full text of the student's summary.
    content - The content score for the summary. The first target.
    wording - The wording score for the summary. The second target.
summaries_test.csv - Summaries in the test set. Contains all fields above except content and wording.

prompts_train.csv - The four training set prompts. Each prompt comprises the complete summarization assignment given to students.

    prompt_id -  The ID of the prompt which links to the summaries file.
    prompt_question -  The specific question the students are asked to respond to.
    prompt_title -  A short-hand title for the prompt.
    prompt_text -  The full prompt text.

prompts_test.csv - The test set prompts. Contains the same fields as above. The prompts here are only an example. The full test set has a large number of prompts. The train / public test / private test splits do not share any prompts.

sample_submission.csv - A submission file in the correct format. See the Evaluation page for details.

Please note that this is a Code Competition. To help you author your submissions, we provide a some example data in summaries_test.csv and prompts_test.csv in the correct format. When your submission is scored, this example test data will be replaced with the full test set. The full test set comprises about 17,000 summaries from a large number of prompts.

prompts_train.csv - column name: prompt_id, prompt_question, prompt_title, prompt_text
summaries_train.csv - column name: student_id, prompt_id, text, content, wording
prompts_test.csv - column name: prompt_id, prompt_question, prompt_title, prompt_text
summaries_test.csv - column name: student_id, prompt_id, text


## Dataset folder Location: 
../../kaggle-data/commonlit-evaluate-student-summaries. In this folder, there are the following files you can use: sample_submission.csv, prompts_train.csv, summaries_train.csv, prompts_test.csv, summaries_test.csv

## Solution Description:
#### Inputs

The data was input to the model as follows :

'Think through this step by step : ' + prompt_question + [SEP] + 'Pay attention to the content and wording : ' + text + [SEP] + prompt_text

#### Pooling Method [High Impact]

Input : [TOKEN] [TOKEN] [SEP] [TOKEN] [TOKEN] [SEP] [TOKEN] [TOKEN]
Head Mask : [0] [0] [1] [1] [1] [0] [0] [0]

Instead of using the normal attention mask created by the model tokenizer. I used a head mask that only had ones for the students' answer (text) portion of the input and zeros for all other tokens. I used the normal attention mask for the attention mask that the model consumed but I used the head mask for the mean pooling.

This had the biggest impact out of all the tricks I used. It increased the CV by a huge margin in all folds, but especially for the difficult prompts : 3b9047 and 814d6b. In my opinion this was the “magic” for this competition.

#### Prompt Question Augmentation [Moderate Impact]

I created 10 extra prompt questions per a prompt. I used an LLM. I asked the LLM to give me 10 variations of the prompt question. I then used this as augmentation during training. In inference, I used the default prompt question. In total I had 44 different prompt questions across all folds.

#### Auxiliary Classes [Moderate Impact]

I used auxiliary classes during the competition. These auxiliary classes were the target classes from Feedback 3.0 -
['cohesion','syntax','vocabulary','phraseology','grammar','conventions'].

To create these labels I used models that were trained on the Feedback 3.0 data and ran the data from this competition through those models. I used only the ‘text’ column from this competition. In doing this I produced pseudo labels to use for this competition.

I used the auxiliary classes in the following way : (loss * .5) + (aux_loss * .5)

The auxiliary classes were used every second step.

The Feedback 3.0 competition was hosted by The Learning Agency Lab and to the best of my knowledge this is a legal technique.

#### Max Length

Models were trained on a maximum length ranging from 896-1280 during initial training. During the pseudo labelling rounds they were trained with a maximum length ranging from 1280-2048. Pseudo labels allowed the models to learn at a higher maximum length.

During inference the models used 1792 for large and 2048 for base.

#### Pseudo Labels [Moderate Impact]

Once a CV of .4581 was reached across the grouped kfold I started creating pseudo labels.

The pseudo labels allowed me to train deberta-v3-base effectively. Before PL, I was not able to train the base model. They also allowed me to increase the maximum length during training.

PL increased the CV from .4581 to .4476

The models were trained using a concatenation of the original labels and pseudo labels.

#### Final Ensemble (PL)

    Model Name	Training Max Length	Inference Max Length	Head	Model CV
    microsoft/deberta-v3-large	2048	1792	Mean Pooling + LSTM Layer Pooling	.460
    microsoft/deberta-v3-base	2048	2048	Mean Pooling + LSTM Sequence Pooling	.468
    OpenAssistant/reward-model-deberta-v3-large-v2	2048	1792	Mean Pooling + LSTM Layer Pooling	.464
    microsoft/deberta-large	2048	1792	Mean Pooling + Linear	.466
    microsoft/deberta-v3-large	1280	1792	Mean Pooling + LSTM Sequence Pooling	.461

#### Did work:

    Layer wise learning rate decay
    Freezing layers (bottom 8)
    LSTM layer pooling
    LSTM Sequence pooling
    Turn off dropout in transformer backbone
    Multisample dropout in head



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: