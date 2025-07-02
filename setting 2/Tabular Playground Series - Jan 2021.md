You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Jan_2021_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions, and thus more beginner-friendly. 

In order to have a more consistent offering of these competitions for our community, we're trying a new experiment in 2021. We'll be launching a month-long tabular Playground competition on the 1st of every month, and continue the experiment as long as there's sufficient interest and participation.

The goal of these competitions is to provide a fun, but less challenging, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals. 

##  Evaluation Metric:
Submissions are scored on the root mean squared error. RMSE is defined as:
$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$
where \( \hat{y} \) is the predicted value, \( y \) is the original value, and \( n \) is the number of rows in the test data.

Submission File

For each row in the test set, you must predict the value of the target as described on the data tab, each on a separate row in the submission file. The file should contain a header and have the following format:

    id,target
    0,0.5
    2,10.2
    6,2.2
    etc.


##  Dataset Description:
For this competition, you will be predicting a continuous target based on a number of feature columns given in the data. All of the feature columns, cont1 - cont14 are continuous.

Files

    train.csv - the training data with the target column
    test.csv - the test set; you will be predicting the target for each row in this file
    sample_submission.csv - a sample submission file in the correct format

train.csv - column name: id, cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, cont14, target
test.csv - column name: id, cont1, cont2, cont3, cont4, cont5, cont6, cont7, cont8, cont9, cont10, cont11, cont12, cont13, cont14


## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-jan-2021. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
#### Overall

My solution is based on stacking diverse models. And I had 4 main models:

    1 - regular lgbm modeling with plain features.
    2 - 2-stage lgbm modeling. For this part, I turned the real target column into a binary column with a threshold and then run a classification model as 1st-stage modeling. Let's say my threshold was 8, then targets having a lower value than 8 were classified as 0 and the rest as 1. Then I got predictions for the probability of being in class 1. At my 2nd-stage modeling, again I had 2 separate models. First, I run my models with training data having the target higher than the threshold and got predictions for both validation and test. Then did the same with the training data having the target values lower than the threshold. In the end, I calculated final predictions as: final_prediction = probability_of_being_0 * (predictions_from_training_0) + probability_of_being_1 * (predictions_from_training_1). Just by changing the threshold value, I managed to get many diverse models easily, and accumulated their oof and test set predictions
    3 - regular lgbm with the augmented training dataset. I tried to apply DAE on all data set. Even though my dae was far from being successful as 1st place's dae solution, using the output of my DAE for augmenting training data during the CV ended up improving final stacking well enough.
    4 - MLP with Embedding layers after crafting categorical features from the original continous columns. By adding embedding layers along with the original inputs I managed to get 0.700X CV and LB with an NN model.

As the final step, I applied stacking with a linear regression on top of OOF predictions of my each saved model. It was much better than manual blending. Also as a very final squeezing step, I've created non-linear interaction features between oof features during stacking and managed to improve both the CV and LB in 4th-5th decimals more.


#### NN
As a first thing, I've created 3 categorical features for each cont feature. 2 of my categorical features are generated by only rounding numerical columns to lower decimals. One with 2 decimals and the other with 1 decimal. It's something like binning. My third categorical feature is obtained via pandas.cut function. I binned the continos features into 10 bins.

I also applied labelencoder to get final categorical feature values.

max_cat_values, max_cat_values3 and max_cat_values4 lists will be used in Embedding layer in the NN model.

The framework of the nn is shown as follows:

1. **Inputs:**
   - The model takes multiple inputs, including categorical and continuous features.
   - For each categorical feature, an `Input` layer is created with a shape of `(1,)` and data type `int32`.
   - There is also a single `Input` layer for continuous features with a shape of `(14,)` and data type `float32`.

2. **Embedding Layers:**
   - For each categorical feature, there are three separate embedding processes, each with different configurations:
     - **First Embedding Layer:**
       - Each categorical feature is embedded into a 3-dimensional space.
       - A `Dropout` layer with a rate of 0.6 is applied to the embeddings.
       - The embeddings are then flattened using a `Flatten` layer.
     - **Second and Third Embedding Layers:**
       - Each categorical feature is embedded into a 12-dimensional space.
       - A `Dropout` layer with a rate of 0.5 is applied to the embeddings.
       - The embeddings are flattened using a `Flatten` layer.

3. **Concatenation and Dense Layers:**
   - The flattened embeddings from the first embedding layer are concatenated with the continuous features.
   - This concatenated output is passed through two `Dense` layers with 1024 units each and ReLU activation. The first `Dense` layer includes L2 regularization.
   - The flattened embeddings from the second and third embedding layers are each concatenated separately and passed through two `Dense` layers with 2048 units each and ReLU activation.

4. **Final Concatenation and Output:**
   - The outputs from the three sets of dense layers (corresponding to the three embedding processes) are concatenated.
   - The concatenated output is passed through a final `Dense` layer with a single unit and a linear activation function to produce the final output.

5. **Compilation:**
   - The model is compiled using the Adam optimizer with a learning rate of 0.001.
   - The loss function used is Mean Squared Error (MSE).

This architecture is designed to handle a mix of categorical and continuous data, leveraging embeddings to transform categorical inputs into dense representations, which are then processed alongside continuous features through dense layers to predict a continuous target variable.

#### LGBM

1 - First model is my main best single model.

2 - Second model is a 2-stage lgbm. As a first thing I turn target column into a binary column and run 1st model to predict the probability of being class 1 for each row. In 2nd stage models, I only use training data with class 1 and get real target predictions for validation and test set. Then doing the same by only using training data with class 0. At then end I calcualte final prediction as like this:  
final_prediction = probability_of_being_0 (predictions_from_training_0) + probability_of_being_1 (predictions_from_training_1)

3 - As a third model, I tried to apply dae. Even though I failed with it (given the 1st place's solution :)), it added diverstiy to my final blending in the end. First I run a simple MLP model with a linear output and then run it for a couple epochs. Later on, I used the MLP output for augmenting the original training data during my CV.






Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: