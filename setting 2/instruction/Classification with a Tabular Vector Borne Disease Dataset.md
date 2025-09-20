You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Classification_with_a_Tabular_Vector_Borne_Disease_Dataset_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Welcome to the 2023 edition of Kaggle's Playground Series!

Thank you to everyone who participated in and contributed to Season 3 Playground Series so far! 
With the same goal to give the Kaggle community a variety of fairly light-weight challenges that can be used to learn and sharpen skills in different aspects of machine learning and data science, we will continue launching the Tabular Tuesday in April every Tuesday 00:00 UTC, with each competition running for 2 weeks. Again, these will be fairly light-weight datasets that are synthetically generated from real-world data, and will provide an opportunity to quickly iterate through various model and feature engineering ideas, create visualizations, etc.. 

Synthetically-Generated Datasets

Using synthetic data for Playground competitions allows us to strike a balance between having real-world data (with named features) and ensuring test labels are not publicly available. This allows us to host competitions with more interesting datasets than in the past. While there are still challenges with synthetic data generation, the state-of-the-art is much better now than when we started the Tabular Playground Series two years ago, and that goal is to produce datasets that have far fewer artifacts. Please feel free to give us feedback on the datasets for the different competitions so that we can continue to improve!

##  Evaluation Metric:
Submissions will be evaluated based on MPA@3. Each submission can contain up to 3 predictions (all separated by spaces), and the earlier a correct prediction occurs, the higher score it will receive.

Submission File

For each id in the test set, you must predict the target prognosis. The file should contain a header and have the following format:

    id,prognosis
    707,Dengue West_Nile_fever Malaria
    708,Lyme_disease West_Nile_fever Dengue
    709,Dengue West_Nile_fever Lyme_disease
    etc.


##  Dataset Description:
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Vector Borne Disease Prediction dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance. Note that in the original dataset some prognoses contain spaces, but in the competition dataset spaces have been replaced with underscores to work with the MPA@K metric.

Files

    train.csv - the training dataset; prognosis is the target
    test.csv - the test dataset; your objective is to predict prognosis
    sample_submission.csv - a sample submission file in the correct format

train.csv - column names: id, prognosis, and some feature names. 
test.csv - column names: id, and some feature names. 



## Dataset folder Location: 
../../kaggle-data/playground-series-s3e13. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
First of all, I would like to thank Kaggle for organising this Playground Series. It was such a nice experience for a competition beginner like me.
To be honest, my main aim in joining this competition was to learn how to use autoencoders within an ensemble model. Practically, it was heavily inspired by this legendary thread. The second aim is to get the merchandise :D.
Therefore, I understand that I did not put a lot of effort into pre-processing the data, and could say that I was very lucky in this competition.
I joined this competition a bit late, so I was only able to produce 4 types of models. No feature engineering, all features being used were scaled using a standard scaler.

    Model Name Public LB Private LB Notes
    LightGBM 0.31677 0.41337 Nothing fancy, just a simple LightGBM model with default parameters
    Neural Network 0.33995 0.44078 Simple NN using 64-64relu-32relu-11softmax as the layers. The first 64 is for the input layer = num of features on the dataset
    Autoencoder 0.37196 0.46052 The autoencoder uses bottleneck architecture 64-64relu-32relu-16relu-32relu-64linear. Take the encoder part (up until the 16relu), freeze it, and add 16relu-11softmax on top of it
    Ensemble 0.35871 0.53179 This is a simple averaging ensemble model from the previous three models. Below this table is the explanation.

Each of the models would be able to generate each class' probability. I thought that maybe by averaging each class' probability on each model (for example, averaging the probability of "Malaria" from LightGBM, Neural Network, and Autoencoder), I can somewhat make an educated guess. 

I assumed that if something is very convincing for most models (let's say, Neural Network and Autoencoder are convinced that the outcome is Malaria), but not that convincing for the other (maybe Dengue for LightGBM), I would still choose the majority vote here.

Therefore, what I did was to average the probabilities and get the top 3 probabilities after averaging them.
My notebook is a mess and clearly lacks any explanation, so I am sorry that I could not post the notebook publicly. 
I was thinking of doing stacking as well but I haven't managed to get my code working, so I stopped here.
I am very open to discussion and would like to hear any thoughts from you. Have a great week ahead!
Also, I would like to thank @belati and @mpwolke . You guys being very active in this competition has encouraged me to keep trying!.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: