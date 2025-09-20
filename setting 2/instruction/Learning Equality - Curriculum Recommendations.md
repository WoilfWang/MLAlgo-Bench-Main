You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Learning_Equality_-_Curriculum_Recommendations_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
#### Goal of the Competition
The goal of this competition is to streamline the process of matching educational content to specific topics in a curriculum. You will develop an accurate and efficient model trained on a library of K-12 educational materials that have been organized into a variety of topic taxonomies. These materials are in diverse languages, and cover a wide range of topics, particularly in STEM (Science, Technology, Engineering, and Mathematics).

Your work will enable students and educators to more readily access relevant educational content to support and supplement learning.

#### Context
Every country in the world has its own educational structure and learning objectives. Most materials are categorized against a single national system or are not organized in a way that facilitates discovery. The process of curriculum alignment, the organization of educational resources to fit standards, is challenging as it varies between country contexts.

Current efforts to align digital materials to national curricula are manual and require time, resources, and curricular expertise, and the process needs to be made more efficient in order to be scalable and sustainable. As new materials become available, they require additional efforts to be realigned, resulting in a never-ending process. There are no current algorithms or other AI interventions that address the resource constraints associated with improving the process of curriculum alignment.

Competition host Learning Equality is committed to enabling every person in the world to realize their right to a quality education, by supporting the creation, adaptation, and distribution of open educational resources, and creating supportive tools for innovative pedagogy. Their core product is Kolibri, an adaptable set of open solutions and tools specially designed to support offline-first teaching and learning for the 37% of the world without Internet access. Their close partner UNHCR has consistently highlighted the strong need and innovation required to create automated alignment tools to ensure refugee learners and teachers are provided with relevant digital learning resources. They have been jointly exploring this challenge in depth for the past few years, engaging with curriculum designers, teachers, and machine learning experts. In addition, Learning Equality is partnering with The Learning Agency Lab, an​ independent nonprofit focused on developing science of learning-based tools and programs for social good, along with UNHCR to engage you in this important process.

You have the opportunity to use your skills in machine learning to support educators and students around the world in accessing aligned learning materials that are relevant for their particular context. Better curriculum alignment processes are especially impactful during the onset of new emergencies or crises, where rapid support is needed, such as for refugee learners, and during school closures as took place during COVID-19.

##  Evaluation Metric:
Submissions will be evaluated based on their mean F2 score. The mean is calculated in a sample-wise fashion, meaning that an F2 score is calculated for every predicted row, then averaged.

Submission File

For each topic_id in the test set, you must predict a space-delimited list of recommended content_ids for that topic. The file should contain a header and have the following format:

    topic_id,content_ids
    t_00004da3a1b2,c_1108dd0c7a5d c_376c5a8eb028 c_5bc0e1e2cba0 c_76231f9d0b5e
    t_00068291e9a4,c_639ea2ef9c95 c_89ce9367be10 c_ac1672cdcd2c c_ebb7fdf10a7e
    t_00069b63a70a,c_11a1dc0bfb99
    ...


##  Dataset Description:
The dataset presented here is drawn from the Kolibri Studio curricular alignment tool, in which users can create their own channel, then build out a topic tree that represents a curriculum taxonomy or other hierarchical structure, and finally organize content items into these topics, by uploading their own content and/or importing existing materials from the Kolibri Content Library of Open Educational Resources.

An example of a branch of a topic tree is: Secondary Education >> Ordinary Level >> Mathematics >> Further Learning >> Activities >> Trigonometry. The leaf topic in this branch might then contain (be correlated with) a content item such as a video entitled Polar Coordinates.

You are challenged to predict which content items are best aligned to a given topic in a topic tree, with the goal of matching the selections made by curricular experts and other users of the Kolibri Studio platform. In other words, your goal is to recommend content items to curators for potential inclusion in a topic, to reduce the time they spend searching for and discovering relevant content to consider including in each topic.

Please note that this is a Code Competition, in which the actual test set is hidden. In this public version, we give some sample data drawn from the training set to help you author your solutions. When your submission is scored, this example test data will be replaced with the full test set.

The full test set includes an additional 10,000 topics (none present in the training set) and a large number of additional content items. The additional content items are only correlated to test set topics.

#### Files and Fields
The training set consists of a corpus of topic trees from within the Kolibri Content Library, along with additional non-public aligned channels, and supplementary channels with less granular or lower-quality alignment.

topics.csv - Contains a row for each topic in the dataset. These topics are organized into "channels", with each channel containing a single "topic tree" (which can be traversed through the "parent" reference). Note that the hidden dataset used for scoring contains additional topics not in the public version. You should only submit predictions for those topics listed in sample_submission.csv.

    id - A unique identifier for this topic.
    title - Title text for this topic.
    description - Description text (may be empty)
    channel - The channel (that is, topic tree) this topic is part of.
    category  - Describes the origin of the topic. 
    
        source - Structure was given by original content creator (e.g. the topic tree as imported from Khan Academy). There are no topics in the test set with this category. 
        aligned - Structure is from a national curriculum or other target taxonomy, with content aligned from multiple sources. 
        supplemental - This is a channel that has to some extent been aligned, but without the same level of granularity or fidelity as an aligned channel.

    language - Language code for the topic. May not always match apparent language of its title or description, but will always match the language of any associated content items.
    parent - The id of the topic that contains this topic, if any. This field if empty if the topic is the root node for its channel.
    level - The depth of this topic within its topic tree. Level 0 means it is a root node (and hence its title is the title of the channel).
    has_content - Whether there are content items correlated with this topic. Most content is correlated with leaf topics, but some non-leaf topics also have content correlations.

content.csv - Contains a row for each content item in the dataset. Note that the hidden dataset used for scoring contains additional content items not in the public version. These additional content items are only correlated to topics in the test set. Some content items may not be correlated with any topic.

    id - A unique identifier for this content item.
    title - Title text for this content item.
    description - Description text. May be empty.
    language - Language code representing the language of this content item.
    kind - Describes what format of content this item represents, as one of:

        document (text is extracted from a PDF or EPUB file)
        video (text is extracted from the subtitle file, if available)
        exercise (text is extracted from questions/answers)
        audio (no text)
        html5 (text is extracted from HTML source)
    text - Extracted text content, if available and if licensing permitted (around half of content items have text content).
    copyright_holder - If text was extracted from the content, indicates the owner of the copyright for that content. Blank for all test set items.
    license - If text was extracted from the content, the license under which that content was made available. Blank for all test set items.

correlations.csv The content items associated to topics in the training set. A single content item may be associated to more than one topic. In each row, we give a topic_id and a list of all associated content_ids. These comprise the targets of the training set.

sample_submission.csv - A submission file in the correct format. See the Evaluation page for more details. You must use this file to identify which topics in the test set require predictions.

correlations.csv - column name: topic_id, content_ids
topics.csv - column name: id, title, description, channel, category, level, language, parent, has_content
content.csv - column name: id, title, description, kind, text, language, copyright_holder, license


## Dataset folder Location: 
../../kaggle-data/learning-equality-curriculum-recommendations. In this folder, there are the following files you can use: correlations.csv, topics.csv, sample_submission.csv, content.csv

## Solution Description:

I had given a long break on Kaggle and came back with this competition because I thought it was very relevant to the app I am building, namely Epicurus. This break made me miss Kaggle and kept me motivated in this competition
My actual solution and efficiency solution are very similar. So I will describe both of them at the same time. I hope I can refactor my code and share on Github soon.

### Pipeline:
Candidate Selection (Retriever methods) -> Feature Engineering -> Lightgbm -> Postprocessing

### Validation Scheme:

    1 fold validation
    All source topics and random 67% of the other topics are selected for the training set. The rest are validation topics.
    The contents which only match with validation topics are excluded from the training set.
    For evaluation, validation topics are matched with all the contents and competition metric is calculated.
    While training lightgbm model on the candidates, group4fold on topic_id is used on the validation set. Evaluation is done on the whole validation set afterwards.

At the end of the competition, I had 0.764 validation score and 0.727 LB. While it is a big gap, improvements in my validation score were almost always correlated with LB. And I got my validation score as my Private LB score, which I didnt expect.

Edit: Efficiency model got 0.718 validation, 0.688 Public LB, 0.740 Private LB and around 20 minutes CPU run-time.

### Topic/Content Representation
Each topic is represented as a text using its title, its description and its ancestor titles up to 3 parents above in the tree. Example: 

'Triangles and polygons @ Space, shape and measurement @ Form 1 @ Malawi Mathematics Syllabus | Learning outcomes: students must be able to solve problems involving angles, triangles and polygons including: types of triangles, calculate the interior and exterior angles of a triangle, different types of polygons, interior angles and sides of a convex polygon, the size and exterior angle of any convex polygon.'

Each content is represented as a text using its title, its kind and its description (its text if it doesn’t have a description). Example:

'Compare multi-digit numbers | exercise | Use your place value skills to practice comparing whole numbers.'

### Candidate Selection

#### TFIDF

Char 4gram TFIDF sparse vectors are created for each language and matched with sparse_dot_topn, which is a package I co-authered (https://github.com/ing-bank/sparse_dot_topn) It works very fast and memory efficient. For each topic, top 20 matches above 1% cosine similarity are retrieved.
Transformer Models
I used paraphrase-multilingual-MiniLM-L12-v2 for efficiency track and ensemble of bert-base-multilingual-uncased, paraphrase-multilingual-mpnet-base-v2 (it is actually a xlm-roberta-base) and xlm-roberta-large for the actual competition. 

    Sequence length: 64. But only the first half of the output is mean pooled for the representation vector. Last half is only fed for context. This worked the best for me.
    Arcface training: Training contents are used as classes. Therefore topics have multiple classes and l1-normalized target vectors. The margin starts with 0.1 and increases linearly to 0.5 at the end of 22 epochs. First 2 and last 2 epochs have significantly lower LR. Arcface class centers are initialized with content vectors extracted from pretrained models.
    Ensemble method: Concatenation after l2 normalization

Edit: Models are re-trained with whole data for submission at the end.

Top 20 matches within the same language contents are retrieved.

In addition, for each topic, its closest train set topic is found and its content matches are retrieved as second degree matches.

#### Matches from Same Title Topics
For each topic, train set topics with the same title are found and their matched contents are retrieved.
#### Matches from Same Representation Text Topics
For each topic, train set topics with the same representation text are found and their matched contents are retrieved.
#### Matches from Same Parent Topics
For each topic, train set topics with the same parent are found and their matched contents are retrieved.
All retrieved topic-content pairs are outer joined.

### Feature Engineering

    tfidf match score
    tfidf match score max by topic id
    tfidf match score min by topic id
    vector cosine distance
    vector cosine distance max by topic id
    vector cosine distance min by topic id
    topic title length
    topic description length
    content title length
    content description length
    content text length
    content same title match count
    content same title match count mean over topic id
    content same representation text match count
    content same representation text match count mean over topic id
    content same parent match count
    content same parent match count mean over topic id
    topic language
    topic category
    topic level
    content kind
    same chapter (number extracted from the text)
    starts same
    is content train
    content max train score
    topic max train score
    is content second degree match

### Lightgbm Model

    Hit or miss classification problem
    Overweight hit (minority) class
    Monotonic constraint and 2x feature contribution on most important feature: vector cosine distance
    2 diverse lightgbms: Excluded features which will potentially have different distribution on real test set in one of the models, vector cosine distance min by topic id. Also used slightly different parameters and kfold seed.

### Postprocess
Postprocessing was very important. Using relative probabilities (gaps with highest matches) and using different conditions for train and test set contents were the key. While matching train set contents was like a classification problem, matching test set contents was like an assignment problem.

Topic-content pairs are included if they have one of the conditions below:

    Content has the best matching probability among other contents for the given topic.
    Content is among the train set contents and has above 5% probability and has less than 25% gap with the highest matching probability in the given topic.
    Content is among the test set contents and has less than 5% gap with the highest matching probability in the given topic.
    Content is among the test set contents and the topic is its best match and its total gap* is less than 55%.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: