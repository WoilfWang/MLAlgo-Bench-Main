You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Instacart_Market_Basket_Analysis_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Whether you shop from meticulously planned grocery lists or let whimsy guide your grazing, our unique food rituals define who we are. Instacart, a grocery ordering and delivery app, aims to make it easy to fill your refrigerator and pantry with your personal favorites and staples when you need them. After selecting products through the Instacart app, personal shoppers review your order and do the in-store shopping and delivery for you.

Instacart’s data science team plays a big part in providing this delightful shopping experience. Currently they use transactional data to develop models that predict which products a user will buy again, try for the first time, or add to their cart next during a session. Recently, Instacart open sourced this data - see their blog post on 3 Million Instacart Orders, Open Sourced.

In this competition, Instacart is challenging the Kaggle community to use this anonymized data on customer orders over time to predict which previously purchased products will be in a user’s next order. They’re not only looking for the best model, Instacart’s also looking for machine learning engineers to grow their team.

Winners of this competition will receive both a cash prize and a fast track through the recruiting process. For more information about exciting opportunities at Instacart, check out their careers page here or e-mail their recruiting team directly at ml.jobs@instacart.com.

##  Evaluation Metric:
Submissions will be evaluated based on their mean F1 score.

Submission File

For each order_id in the test set, you should predict a space-delimited list of product_ids for that order. If you wish to predict an empty order, you should submit an explicit 'None' value. You may combine 'None' with product_ids. The spelling of 'None' is case sensitive in the scoring metric. The file should have a header and look like the following:

    order_id,products  
    17,1 2  
    34,None  
    137,1 2 3  
    etc.


##  Dataset Description:
The dataset for this competition is a relational set of files describing customers' orders over time. The goal of the competition is to predict which products will be in a user's next order. The dataset is anonymized and contains a sample of over 3 million grocery orders from more than 200,000 Instacart users. For each user, we provide between 4 and 100 of their orders, with the sequence of products purchased in each order. We also provide the week and hour of day the order was placed, and a relative measure of time between orders. For more information, see the blog post accompanying its public release.

#### File descriptions
Each entity (customer, product, order, aisle, etc.) has an associated unique id. Most of the files and variable names should be self-explanatory.

aisles.csv

    aisle_id,aisle  
    1,prepared soups salads  
    2,specialty cheeses  
    3,energy granola bars  
    ...

departments.csv

    department_id,department  
    1,frozen  
    2,other  
    3,bakery  
    ...

order_products__*.csv

These files specify which products were purchased in each order. order_products__prior.csv contains previous order contents for all customers. 'reordered' indicates that the customer has a previous order that contains the product. Note that some orders will have no reordered items. You may predict an explicit 'None' value for orders with no reordered items. See the evaluation page for full details.

    order_id,product_id,add_to_cart_order,reordered  
    1,49302,1,1  
    1,11109,2,1  
    1,10246,3,0  
    … 
orders.csv

This file tells to which set (prior, train, test) an order belongs. You are predicting reordered items only for the test set orders. 'order_dow' is the day of week.

    order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order  
    2539329,1,prior,1,2,08,  
    2398795,1,prior,2,3,07,15.0  
    473747,1,prior,3,3,12,21.0  
    …

products.csv

    product_id,product_name,aisle_id,department_id
    1,Chocolate Sandwich Cookies,61,19  
    2,All-Seasons Salt,104,13  
    3,Robust Golden Unsweetened Oolong Tea,94,7  
    ...

sample_submission.csv

    order_id,products
    17,39276  
    34,39276  
    137,39276  
    …

aisles.csv - column name: aisle_id, aisle
orders.csv - column name: order_id, user_id, eval_set, order_number, order_dow, order_hour_of_day, days_since_prior_order
order_products__prior.csv - column name: order_id, product_id, add_to_cart_order, reordered
products.csv - column name: product_id, product_name, aisle_id, department_id
departments.csv - column name: department_id, department
order_products__train.csv - column name: order_id, product_id, add_to_cart_order, reordered


## Dataset folder Location: 
../../kaggle-data/instacart-market-basket-analysis. In this folder, there are the following files you can use: sample_submission.csv, aisles.csv, orders.csv, order_products__prior.csv, products.csv, departments.csv, order_products__train.csv

## Solution Description:
The task was reformulated as a binary prediction task: Given a user, a product, and the user's prior purchase history, predict whether or not the given product will be reordered in the user's next order. In short, the approach was to fit a variety of generative models to the prior data and use the internal representations from these models as features to second-level models.

### First-level models
The first-level models vary in their inputs, architectures, and objectives, resulting in a diverse set of representations.

##### Product RNN/CNN
a combined RNN and CNN trained to predict the probability that a user will order a product at each timestep. The RNN is a single-layer LSTM and the CNN is a 6-layer causal CNN with dilated convolutions.

##### Aisle RNN
an RNN similar to the first model, but trained at the aisle level (predict whether a user purchases any products from a given aisle at each timestep).

##### Department RNN
an RNN trained at the department level. The architecture is:

	1.	Input Data:
	•	The input includes user-specific data, department data, and historical order data (e.g., day of week, hour, size, days since last order, etc.).
	•	Sequence data is represented in a 2D format ([batch_size, sequence_length]) and converted into one-hot encoded or scaled formats.
	2.	Embeddings:
	•	User Embeddings: A trainable embedding for users is learned, represented as a dense vector.
	•	Department Data: One-hot encoded and tiled across the sequence length.
	3.	Sequence Features:
	•	Historical data such as “order day of the week” and “order size” are one-hot encoded.
	•	Scalar features are derived by normalizing certain features (e.g., dividing order day by 7).
	•	All historical features are concatenated into a unified tensor.
	4.	Model Architecture:
	•	Input Concatenation: Combines historical features, user embeddings, and department data into a single tensor.
	•	LSTM Layer: Processes the concatenated input using an LSTM with a hidden size of 300. This extracts temporal dependencies from the sequential data.
	•	Dense Layers:
	•	A time-distributed dense layer applies ReLU activation to the LSTM output.
	•	Another time-distributed dense layer predicts the probability of the next item being ordered using a sigmoid activation function.
	5.	Output:
	•	Predictions (y_hat): Probabilities for each time step in the sequence.
	•	Final States and Predictions: The last relevant state and prediction for each sequence are extracted based on the sequence lengths.

##### Product RNN mixture model
an RNN similar to the first model, but instead trained to maximize the likelihood of a bernoulli mixture model.

##### Order size RNN
an RNN trained to predict the next order size, minimizing RMSE.

##### Order size RNN mixture model
an RNN trained to predict the next order size, maximizing the likelihood of a gaussian mixture model.

##### Skip-Gram with Negative Sampling (SGNS)
SGNS trained on sequences of ordered products.

##### Non-Negative Matrix Factorization (NNMF)
NNMF trained on a matrix of user-product order counts.

### Second-level models
The second-level models use the internal representations from the first-level models as features.

    GBM (code): a lightgbm model.
    Feedforward NN (code): a feedforward neural network.

The final reorder probabilities are a weighted average of the outputs from the second-level models. The final basket is chosen by using these probabilities and choosing the product subset with maximum expected F1-score.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: