You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_May_2022_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
The May edition of the 2022 Tabular Playground series binary classification problem that includes a number of different feature interactions. This competition is an opportunity to explore various methods for identifying and exploiting these feature interactions.

About the Tabular Playground Series

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.

The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model. These competitions are a great choice for people looking for something in between the Titanic Getting Started competition and the Featured competitions. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you; thus, we encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.

Getting Started

For ideas on how to improve your score, check out the Intro to Machine Learning and Intermediate Machine Learning courses on Kaggle Learn.

We've also built a starter notebook for you that uses TensorFlow Decision Forests, a TensorFlow library that matches the power of XGBoost with a friendly, straightforward user interface.


##  Evaluation Metric:
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

Submission File

For each id in the test set, you must predict a probability for the target variable. The file should contain a header and have the following format:

    id,target
    900000,0.65
    900001,0.97
    900002,0.02
    etc.


##  Dataset Description:

For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.
Good luck!

Files

    train.csv - the training data, which includes normalized continuous data and categorical data
    test.csv - the test set; your task is to predict binary target variable which represents the state of a manufacturing process
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_27, f_28, f_29, f_30, target
test.csv - column name: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_27, f_28, f_29, f_30


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-may-2022. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
My solution is somewhat simpler, I used a Catboost Langevin model with a depth of 8 and a huge regularization coefficient (this helps the training not get stuck in a dead point).

Then I combined CatBoost model results with the Keras solution…

The model is shown as:
The network structure in the image can be described as a deep feedforward neural network with the following architecture:
	1.	Input Layer:
	•	The input layer takes input with a shape of (None, 44), where None represents the batch size, and 44 is the number of input features.
	2.	Dense Layer 1:
	•	A fully connected (Dense) layer with 64 units. The input is (None, 44), and the output is (None, 64).
	3.	Batch Normalization 1:
	•	A batch normalization layer applied to the output of the first dense layer. Input and output shapes are (None, 64).
	4.	Dense Layer 2:
	•	Another Dense layer with 32 units. Input is (None, 64), and the output is (None, 32).
	5.	Batch Normalization 2:
	•	A batch normalization layer applied to the output of the second dense layer. Input and output shapes are (None, 32).
	6.	Dense Layer 3:
	•	A Dense layer with 32 units. Input and output shapes are (None, 32).
	7.	Batch Normalization 3:
	•	Batch normalization is applied to the output of the third dense layer. Input and output shapes are (None, 32).
	8.	Dense Layer 4:
	•	A Dense layer with 16 units. Input is (None, 32), and the output is (None, 16).
	9.	Batch Normalization 4:
	•	Batch normalization is applied to the output of the fourth dense layer. Input and output shapes are (None, 16).
	10.	Dense Layer 5:
	•	A Dense layer with 8 units. Input is (None, 16), and the output is (None, 8).
	11.	Batch Normalization 5:
	•	Batch normalization is applied to the output of the fifth dense layer. Input and output shapes are (None, 8).
	12.	Output Layer:
	•	A final Dense layer with 1 unit. This is likely used for regression or binary classification. The input is (None, 8), and the output is (None, 1).

Key Characteristics:
	•	Layer Types: Alternates between fully connected (Dense) layers and batch normalization layers.
	•	Reducing Dimensionality: The number of units in the Dense layers decreases progressively from 64 to 1, likely aiming to compress and process the input features.
	•	Batch Normalization: Improves training stability and speeds up convergence by normalizing activations.

I used a modified Mish activation function, I made the activation more sensitive (be careful with it, it often gives a gradient vanishing):

```pyhon
def custom_mish(x):
    return tf.keras.layers.Lambda(lambda x: x*tf.keras.backend.tanh(1+tf.keras.backend.log(tf.keras.backend.exp(x*0.7978845608028654))))(x)
```

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: