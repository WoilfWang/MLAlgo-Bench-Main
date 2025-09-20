You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Tabular_Playground_Series_-_Jun_2022_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
The June edition of the 2022 Tabular Playground series is all about data imputation. The dataset has similarities to the May 2022 Tabular Playground, except that there are no targets. Rather, there are missing data values in the dataset, and your task is to predict what these values should be.

About the Tabular Playground Series

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.

The goal of these competitions is to provide a fun and approachable-for-anyone tabular dataset to model. These competitions are a great choice for people looking for something in between the Titanic Getting Started competition and the Featured competitions. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you; thus, we encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.

##  Evaluation Metric:
Submissions are scored on the root mean squared error. RMSE is defined as:
$$\textrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
where $\hat{y}_i$ is the predicted value and yi is the original value for each instance i.

Submission File

For each row-col pair in the in the sample_submission.csv file (corresponding to all of the missing values found in data.csv), you must predict the missing value of that data point. The file should contain a header and have the following format:

    row-col, value
    0-F_1_14, 1.54
    0-F_3_23, -0.56
    1-F_3_24, 0.01
    etc.


##  Dataset Description:
For this challenge, you are given (simulated) manufacturing control data that contains missing values due to electronic errors. Your task is to predict the values of all missing data in this dataset. (Note, while there are continuous and categorical features, only the continuous features have missing values.)
Here's a notebook that you can use to get started.

Files

    data.csv - the file includes normalized continuous data and categorical data; your task is to predict the values of the missing data.
    sample_submission.csv - a sample submission file in the correct format; the row-col indicator corresponds to the row and column of each missing value in data.csv

data.csv - column names: row_id, F_1_0, F_1_1, ..., F_1_14, F_2_0, ..., F_2_24, F_3_0, ..., F_3_24, F_4_0, ..., F_4_14. 

## Dataset folder Location: 
../../kaggle-data/tabular-playground-series-jun-2022. In this folder, there are the following files you can use: sample_submission.csv, data.csv

## Solution Description:
As most people noticed during this competition, the significant challenge was estimating the conditional distribution of F4 where two or more values were missing in the given row. To solve this, I used a denoising autoencoder, which estimates the distribution of missing values given a mask of where the values are missing. I mean-imputed F1 and F3, and completely ignored F2.

A pytorch implementation of my notebook can be found here  (it scored poorly because I messed up the submission dataframe 😅).

Below, I have included a drawing of this architecture.


1. Input Data:
	•	The model begins with a dataset that may contain missing values.

2. Preprocessing:
	•	Null Imputation (X): All null values in the dataset are initially imputed with zeros (X(0 impute nulls)).
	•	Source Missing Indicator (source_nan_dummy): A binary mask is created where a value of 1 indicates that the corresponding feature in the original data is missing (data.isnull), and 0 otherwise.

3. Masking:
	•	Random Binomial Mask: A random dropout mask is generated using a binomial distribution to simulate additional missingness during training. This mask is applied to X, creating x_mi (masked input).
	•	DAE Input Mask (m): This is a union of the original missing values and the random dropout mask, identifying all missing elements in the current input.

4. Feature and Mask Embeddings:
	•	Feature Linear Embedding: Each feature in x_mi is linearly embedded into a representation suitable for the downstream DAE model.
	•	Mask Linear Embedding: Similarly, the mask (m) is embedded linearly to encode the information about missing values.

5. Denoising Autoencoder (DAE):
	•	The embeddings of the features and masks are input into the DAE.
	•	Add + Layernorm: The two embeddings are added together and normalized using LayerNorm.
	•	Flatten: The resulting representations are flattened to prepare for downstream processing.

6. Multi-Layer Perceptron (MLP):
	•	The flattened output passes through a Multi-Layer Perceptron (MLP) with skip connections and LayerNorm for enhanced learning and stability.
	•	The MLP is repeated for 7 layers (x7), adding depth to the model to learn complex patterns.

7. Output Prediction:
	•	The output dimension of the MLP is reduced to match the required output size using a linear layer (Linear (output_dim)).

8. Output Reconstruction:
	•	The final predicted values (x_pred) are combined with the original masked values (x_m) to reconstruct the input. This is done using the formula:
￼
	•	This ensures that the predicted values are used only for the missing elements, while the original values are retained for the known elements.

1. Loss Function:
	•	Masked Mean Squared Error (MSE) Loss: The model uses a loss function that computes MSE only for the missing elements (i.e., where m = 1). The loss is zero for the known elements (m = 0), ensuring the model focuses on imputing the missing values.

#### Random Mask

I initially impute the data with zeros, and create a source null matrix that contains the locations of the original data nulls. Then I create a binomial random mask (where each row has at least one value) that I multiply by the original data to randomly set values to zero. The model takes the masked data and an input mask - the input mask is the combination of the source null and random mask vectors.
#### Feature-wise embeddings
The model embeds both the features and the mask and adds them together. This is so it can learn a representation of where the nulls are set to zero, and therefore where to focus to impute. This method performs much better than simply applying dropout to the masked data. For these embeddings, I linearly projected each feature in the input and mask vectors into an embedding dimension (16 in the final submission), added them together, and then flattened them.

I also tried dot-product attention between the mask and linear embeddings (as per this paper), but this performed worse than simply adding them together.
#### MLP Architecture
The feature/mask embeddings are then sent through an MLP. I used layer normalization with skip connections, with 7 dense layers using the mish activation function. The performance increased with larger layer sizes, but I only tried up to a size of 2048, since computation time became a problem and I was having to decrease my batch size dramatically.
#### Output Computation
I send the MLP output through a final dense layer to match the dimensions with the input shape. Then, the final output is conditional on whether the value was masked. If the value was masked, we use the final dense layer output, otherwise we just use the model input. This causes the gradients for non-masked inputs to equal zero with the MSE loss, so the network only updates parameters for their contribution to the imputation of the mask. Like the mask embeddings, this helps the network focus on the values that should be imputed.
#### Masked MSE Loss
This step is very similar to the output  computation, and is just another way of conditionally setting the loss function to zero. Here, I used an MSE loss that sets the value to zero according to a mask. This mask is the true data null values - thus we never calculate a loss on values that were originally in the training data, but we can still learn from those rows without creating bias by using a naive imputation method during training. A similar method is used here (equivalent to setting the loss to zero at the data nulls). This step could have been included as part of the modified output computation, but I only wanted to include this step during training - otherwise the implementations would have to pass a vector of zeros during the prediction step (which felt a bit clunky).
#### Conditional Ensemble
I ran the DAE 3 times in pytorch and tensorflow - tensorflow runs performed significantly better, but the average of all performed the best. This model scored 0.83351 on the private LB. To improve this slightly, I ensembled single-attribute prediction runs only when the row-wise null count for F4 was 1. This conditional ensemble gave the private LB score of 0.83343..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: