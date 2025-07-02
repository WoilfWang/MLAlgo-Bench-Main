You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Mercari_Price_Suggestion_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
It can be hard to know how much something’s really worth. Small details can mean big differences in pricing. For example, one of these sweaters cost $335 and the other cost $9.99. Can you guess which one’s which?

Product pricing gets even harder at scale, considering just how many products are sold online. Clothing has strong seasonal pricing trends and is heavily influenced by brand names, while electronics have fluctuating prices based on product specs.

Mercari, Japan’s biggest community-powered shopping app, knows this problem deeply. They’d like to offer pricing suggestions to sellers, but this is tough because their sellers are enabled to put just about anything, or any bundle of things, on Mercari's marketplace.

In this competition, Mercari’s challenging you to build an algorithm that automatically suggests the right product prices. You’ll be provided user-inputted text descriptions of their products, including details like product category name, brand name, and item condition.

Note that, because of the public nature of this data, this competition is a “Kernels Only” competition. In the second stage of the challenge, files will only be available through Kernels and you will not be able to modify your approach in response to new data. Read more details in the data tab and Kernels FAQ page.

##  Evaluation Metric:
The evaluation metric for this competition is Root Mean Squared Logarithmic Error.
The RMSLE is calculated as
$$\epsilon = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$$
Where:

    \\(\epsilon\\) is the RMSLE value (score)
    \\(n\\) is the total number of observations in the (public/private) data set,
    \\(p_i\\) is your prediction of price, and
    \\(a_i\\) is the actual sale price for \\(i\\). 
    \\(\log(x)\\) is the natural logarithm of \\(x\\)

Submission File

For every row in the dataset, submission files should contain two columns: test_id and price.  The id corresponds to the column of that id in the test.tsv. The file should contain a header and have the following format:

    test_id,price
    0,1.50
    1,50
    2,500
    3,100
    etc.

##  Dataset Description:
In this competition, you will predict the sale price of a listing based on information a user provides for this listing. This is a Kernels-only competition, the files in this Data section are downloadable just for your reference in Stage 1. Stage 2 files will only be available in Kernels and not available for download here. 

### Data fields
#### train.tsv, test.tsv
The files consist of a list of product listings. These files are tab-delimited. 

    train_id or test_id - the id of the listing
    name - the title of the listing. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]
    item_condition_id - the condition of the items provided by the seller
    category_name - category of the listing
    brand_name
    price - the price that the item was sold for. This is the target variable that you will predict. The unit is USD. This column doesn't exist in test.tsv since that is what you will predict. 
    shipping - 1 if shipping fee is paid by seller and 0 by buyer
    item_description - the full description of the item. Note that we have cleaned the data to remove text that look like prices (e.g. $20) to avoid leakage. These removed prices are represented as [rm]

Please note that in stage 1, all the test data will be calculated on the public leaderboard. In stage 2, we will swap the test.tsv file to the complete test dataset that includes the private leaderboard data. 

#### sample_submission.csv
A sample submission file in the correct format. 

    test_id - matches the test_id column in test.tsv
    price

## Dataset folder Location: 
../../kaggle-data/mercari-price-suggestion-challenge. In this folder, there are the following files you can use: sample_submission.csv, train.tsv, test.tsv

## Solution Description:
As a preprocessing step, we kept maximum 60 words for all names and descriptions and apply the same cleaning process on all the texts
We labelEncoded all brand and categories and concatenated the encoded values with the name feature.

We made 4 models :

    1) Ridge model : trained on 1-ngrams and custom bigrams.
    1-ngrams is using both name and description. Our custom bigrams work as follows : we concatenate name with the first 5 words from description, then apply np.unique() on the list of word to sort them and remove duplicates, then create all 2-way possible combination of words.
    The ridge model is able to score .418 on public LB

    2) Sparse NN model trained on a CountVectorizer with ngram_range=(1,2). The NN is fit with sparse data from name and description features

    3) a fastText NN model with a shared_embedding layer for name and description features

    4) another Sparse NN model fit with character Ngrams using name and description features.
   
The framework of the sparse NN is as follows:

This model is a fully connected neural network with multiple layers. Here’s a concise description of its architecture:
	1.	Input Layer: Accepts features with num_features dimensions.
	2.	Hidden Layers:
	•	Layer 1:
	•	Batch normalization applied to the input.
	•	Fully connected layer with 1500 units.
	•	Activation function: Leaky ReLU.
	•	Layer 2:
	•	Batch normalization.
	•	Dropout (50%).
	•	Fully connected layer with 1250 units.
	•	Activation function: Leaky ReLU.
	•	Layer 3:
	•	Batch normalization.
	•	Dropout (35%).
	•	Fully connected layer with 1000 units.
	•	Activation function: Leaky ReLU.
	•	Layer 4:
	•	Batch normalization.
	•	Dropout (30%).
	•	Fully connected layer with 750 units.
	•	Activation function: Leaky ReLU.
	3.	Output Layer:
	•	Batch normalization.
	•	Dropout (25%).
	•	Fully connected layer with num_targets units using weight normalization.
	4.	Activation Functions:
	•	Leaky ReLU activation is used after each hidden layer.
	5.	Dropout Regularization:
	•	Dropout rates are progressively reduced across layers: 50%, 35%, 30%, 25%.

We waste half our time building an LGB model that didnt help even though it was our best model. We found that using different models with the same data representation is not going to work. We then choose to use different NN with different inputs

To make our NN fast, we double the batch size after each epoch. This is quite similar to reducing the learning rate after each epoch with the advantage of speed gain


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: