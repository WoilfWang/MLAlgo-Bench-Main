You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Allstate_Purchase_Prediction_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
As a customer shops an insurance policy, he/she will receive a number of quotes with different coverage options before purchasing a plan. This is represented in this challenge as a series of rows that include a customer ID, information about the customer, information about the quoted policy, and the cost. Your task is to predict the purchased coverage options using a limited subset of the total interaction history. If the eventual purchase can be predicted sooner in the shopping window, the quoting process is shortened and the issuer is less likely to lose the customer's business.
Using a customer’s shopping history, can you predict what policy they will end up choosing?

##  Evaluation Metric:
Submissions are evaluated on an all-or-none accuracy basis. You must predict every coverage option correctly to receive credit for a given customer. Your score is the percent of customers for whom you predict the exact purchased policy.
Submission File
The submission format is created by concatenating each plan option (A,B,C,D,E,F,G) as a single string, in order. The file should contain a header and have the following format:
customer_ID,plan10000001,111111110000002,111111110000003,1111111...

##  Dataset Description:
Files
The training and test sets contain transaction history for customers that ended up purchasing a policy. For each customer_ID, you are given their quote history. In the training set you have the entire quote history, the last row of which contains the coverage options they purchased. In the test set, you have only a partial history of the quotes and do not have the purchased coverage options. These are truncated to certain lengths to simulate making predictions with less history (higher uncertainty) or more history (lower uncertainty).
For each customer_ID in the test set, you must predict the seven coverage options they end up purchasing.
What is a customer?
Each customer has many shopping points, where a shopping point is defined by a customer with certain characteristics viewing a product and its associated cost at a particular time.

Some customer characteristics may change over time (e.g. as the customer changes or provides new information), and the cost depends on both the product and the customer characteristics.
A customer may represent a collection of people, as policies can cover more than one person.
A customer may purchase a product that was not viewed!

### Product Options
Each product has 7 customizable options selected by customers, each with 2, 3, or 4 ordinal values possible:

    | Option name | Possible values |
    | A | 0, 1, 2|
    | B | 0, 1|
    | C | 1, 2, 3, 4 |
    | D | 1, 2, 3|
    | E | 0, 1|
    | F| 0, 1, 2, 3 |
    | G| 1, 2, 3, 4|

A product is simply a vector with length 7 whose values are chosen from each of the options listed above. The cost of a product is a function of both the product options and customer characteristics.

### Variable Descriptions
    customer_ID - A unique identifier for the customer
    shopping_pt - Unique identifier for the shopping point of a given customer
    record_type - 0=shopping point, 1=purchase point
    day - Day of the week (0-6, 0=Monday)
    time - Time of day (HH:MM)
    state - State where shopping point occurred
    location - Location ID where shopping point occurred
    group_size - How many people will be covered under the policy (1, 2, 3 or 4)
    homeowner - Whether the customer owns a home or not (0=no, 1=yes)
    car_age - Age of the customer’s car
    car_value - How valuable was the customer’s car when new
    risk_factor - An ordinal assessment of how risky the customer is (1, 2, 3, 4)
    age_oldest - Age of the oldest person in customer's group
    age_youngest - Age of the youngest person in customer’s group
    married_couple - Does the customer group contain a married couple (0=no, 1=yes)
    C_previous - What the customer formerly had or currently has for product option C (0=nothing, 1, 2, 3,4)
    duration_previous -  how long (in years) the customer was covered by their previous issuer
    A,B,C,D,E,F,G - the coverage options
    cost - cost of the quoted coverage options

train.csv - column name: customer_ID, shopping_pt, record_type, day, time, state, location, group_size, homeowner, car_age, car_value, risk_factor, age_oldest, age_youngest, married_couple, C_previous, duration_previous, A, B, C, D, E, F, G, cost
test.csv - column name: customer_ID, shopping_pt, record_type, day, time, state, location, group_size, homeowner, car_age, car_value, risk_factor, age_oldest, age_youngest, married_couple, C_previous, duration_previous, A, B, C, D, E, F, G, cost


## Dataset folder Location: 
../../kaggle-data/allstate-purchase-prediction-challenge. In this folder, there are the following files you can use: train.csv, sampleSubmission.csv, test.csv

## Solution Description:
The main idea is that we don't know at what shopping_pt the purchase will be made. We know which plans fit more each profiles, so basically I'm training the model on the whole dataset using the purchased plan as target at each shopping_pt during the transaction history for all the customers. Even though purchases never happen at shopping_pt #1, it is included in the training data. The main reason is because patterns which occur at shopping_pt #1 for some customer can occur at different shopping_pt for others customers, leading to the same plan purchased.

I’m using is a Random Forest (scikit-learn implementation) as base model, which by itself only can give quite good results. To produce a robust model, I’ve ensemble 9 Random Forest out of other 50. If five out nine models agree on the same plan then this change is made, otherwise the last quote is used (majority vote).

The final ensemble, which led our team to place 2nd in the private leader board, is the combination of my predicted G and Steve’s ABCEDF.

### Extra Features
I’ve used all the features provided, at exception of date & time. To help tree interaction and improve the accuracy I’ve also included the following features, group by category for your convenience.

Category Interactions (2-way)

    G & shopping_pt ** 1st most important
    G & state ** 7th most important
    state & shopping_pt
    Category & Interaction mapped at arithmetic mean of the cost

    mean of cost grouped by G ** 3rd most important
    mean of cost grouped by State & G
    mean of cost grouped by State
    Average of target variable

    Average of purchased G plus some randomness, grouped by location ** 5th most important
    Average of purchased G plus some randomness, grouped by state ** 6th most important
    Continuous Interactions

    cost / group_size
    cost / car_age
    Naming Convention
    Product: A, B, C, D, E, F and G are all products.
    Plan: Combination of A, B, C, D, E, F and G.
    Baseline: Is the last plan or product quoted at the latest shopping_pt available.

### Metric
The score I’ve used to determine how good a model is defined as follow. The difference between the baseline accuracy and the model accuracy measured at each single shopping_pt, times the number of samples in the test set at that shopping_pt. For example, the difference between the model and the baseline for shopping_pt #2 is 0.4160-0.4116=0.0044 times the count of the test samples where the latest shopping_pt available is #2 is 0.0044x18,943=58.5. I’ll be addressing to score at the sum product of these differences and the test set distribution.

### Modelling Techniques & Training
A Random Forest is by itself an ensemble of decision trees. Each decision tree in a Random Forest is trained on different subset of data, leading to many different trees. The Random Forest predictive power comes with the ensemble of all these trees, stacking the class probabilities.

The higher is the number of tree we’ll build, the more accurate and more stable the prediction will be. This is what happens usually, but not in this problem since the gain over the baseline is very low. Making this more sensible to randomness and harder to fix only increasing the number of tree!

Instead of keep stacking the class probabilities and increasing the number of tree, we can keep the number of tree in the Random Forest lower and look at the output at a number of Random Forests. If the majority of these agree on the same outcome, then is quite likely that change is actually occurring. If the majority have the same outcome, then chose this as final prediction. Otherwise use the safest option: the baseline. Making this strategy is less prone to randomness.

I’ve trained several Random Forest using the same data but using different seed, which approximately lead to ~300 different predictions out of 55,716. Is quite a low number, but in this particular competition one more accurate prediction is the difference between the 2nd and 3rd place! Hence here comes the need to have not just a good model, but a very stable model which will generalize as much constantly as it could on unseen data.

Using the majority vote ideas gave a quite stable prediction (and more accurate). What helped a little bit further was selecting a subset of all the Random Forests which are expected to have a better accuracy. How? While looking for a way to identify more accurate Random Forest I’ve noticed that for higher train set scores, usually there is an higher cross validation score. Following this intuition I could discard model whom their train set score was worse than the others as they are more likely to be not good as the others. Do a majority vote on the best 9 Random Forests instead of using all the 50 improved the results too!



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: