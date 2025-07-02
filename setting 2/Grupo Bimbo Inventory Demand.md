You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Grupo_Bimbo_Inventory_Demand_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Planning a celebration is a balancing act of preparing just enough food to go around without being stuck eating the same leftovers for the next week. The key is anticipating how many guests will come. Grupo Bimbo must weigh similar considerations as it strives to meet daily consumer demand for fresh bakery products on the shelves of over 1 million stores along its 45,000 routes across Mexico.

Currently, daily inventory calculations are performed by direct delivery sales employees who must single-handedly predict the forces of supply, demand, and hunger based on their personal experiences with each store. With some breads carrying a one week shelf life, the acceptable margin for error is small.

In this competition, Grupo Bimbo invites Kagglers to develop a model to accurately forecast inventory demand based on historical sales data. Doing so will make sure consumers of its over 100 bakery products aren’t staring at empty shelves, while also reducing the amount spent on refunds to store owners with surplus product unfit for sale.

##  Evaluation Metric:
The evaluation metric for this competition is Root Mean Squared Logarithmic Error.
The RMSLE is calculated as
$$\epsilon = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$$
Where:

    \\(\epsilon\\) is the RMSLE value (score)
    \\(n\\) is the total number of observations in the (public/private) data set,
    \\(p_i\\) is your prediction of demand, and
    \\(a_i\\) is the actual demand for \\(i\\). 
    \\(\log(x)\\) is the natural logarithm of \\(x\\)

Submission File

For every row in the dataset, submission files should contain two columns: id and Demanda_uni_equi.  The id corresponds to the column of that id in the test.csv. The file should contain a header and have the following format:

    id,Demanda_uni_equil
    0,1
    1,0
    2,500
    3,100
    etc.

##  Dataset Description:
In this competition, you will forecast the demand of a product for a given week, at a particular store. The dataset you are given consists of 9 weeks of sales transactions in Mexico. Every week, there are delivery trucks that deliver products to the vendors. Each transaction consists of sales and returns. Returns are the products that are unsold and expired. The demand for a product in a certain week is defined as the sales this week subtracted by the return next week.

The train and test dataset are split based on time, as well as the public and private leaderboard dataset split.

Things to note:

    There may be products in the test set that don't exist in the train set. This is the expected behavior of inventory data, since there are new products being sold all the time. Your model should be able to accommodate this.
    There are duplicate Cliente_ID's in cliente_tabla, which means one Cliente_ID may have multiple NombreCliente that are very similar. This is due to the NombreCliente being noisy and not standardized in the raw data, so it is up to you to decide how to clean up and use this information. 
    The adjusted demand (Demanda_uni_equil) is always >= 0 since demand should be either 0 or a positive value. The reason that Venta_uni_hoy - Dev_uni_proxima sometimes has negative values is that the returns records sometimes carry over a few weeks.

File descriptions

    train.csv — the training set
    test.csv — the test set
    sample_submission.csv — a sample submission file in the correct format
    cliente_tabla.csv — client names (can be joined with train/test on Cliente_ID)
    producto_tabla.csv — product names (can be joined with train/test on Producto_ID)
    town_state.csv — town and state (can be joined with train/test on Agencia_ID)

Data fields

    Semana — Week number (From Thursday to Wednesday)
    Agencia_ID — Sales Depot ID
    Canal_ID — Sales Channel ID
    Ruta_SAK — Route ID (Several routes = Sales Depot)
    Cliente_ID — Client ID
    NombreCliente — Client name
    Producto_ID — Product ID
    NombreProducto — Product Name
    Venta_uni_hoy — Sales unit this week (integer)
    Venta_hoy — Sales this week (unit: pesos)
    Dev_uni_proxima — Returns unit next week (integer)
    Dev_proxima — Returns next week (unit: pesos)
    Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict)

train.csv - column name: Semana, Agencia_ID, Canal_ID, Ruta_SAK, Cliente_ID, Producto_ID, Venta_uni_hoy, Venta_hoy, Dev_uni_proxima, Dev_proxima, Demanda_uni_equil
test.csv - column name: id, Semana, Agencia_ID, Canal_ID, Ruta_SAK, Cliente_ID, Producto_ID
producto_tabla.csv - column name: Producto_ID, NombreProducto
town_state.csv - column name: Agencia_ID, Town, State
cliente_tabla.csv - column name: Cliente_ID, NombreCliente


## Dataset folder Location: 
../../kaggle-data/grupo-bimbo-inventory-demand. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv, producto_tabla.csv, town_state.csv, cliente_tabla.csv

## Solution Description:
#### Summary

Overall solution was 2nd-level ensembling. We built a lot of models on 1st level (using 9 week as validation set) The training method of most of 1st level models was XGBoost.  For second level we used ExtraTrees classifier and linear model from Python scikit-learn. The final result is weighted average of these two models.

The most important features are based on the 1-3 weeks lags of target variable grouped by factors and their combinations,  aggregated features (min, max, mean, sum) of target variable grouped by factors and their combinations, frequency features of factors variables.

For our work, we used R environment with Rstudio IDE, and Python with Jupyter. We combined our work on our personal computers with cloud computing on x1.32xlarge Amazon EC2 instance with web interface for access. Last week of the competition was the continuous calculation of single and 2nd  level models on x1.32xlarge Amazon EC2 instance.

It was a very interesting competition, our approaches were inspired by many discussions on the competition forum and by many kernels published by other participants.

#### Features Selection / Engineering
One of the main ideas in our approach is that it is important to know what were the previous weeks sales. If previous week too many products were supplied and they were not sold, next week this product amount, supplied to the same store, will be decreased. So, it is very important to include lagged values of target variable Demanda_uni_equil as a feature to predict next sales. 

The simplified version of the R script “Bimbo XGBoost R script LB:0.457” for such an approach was published on the Kaggle forum of this competition. 

 We merged transactions data frame with data frames from files cliente_tabla.csv, producto_tabla.csv using appropriate key fields. We used frequencies features grouped by categorical variables (e.g. Producto_ID, Cliente_ID, Agencia_ID, etc. ) and by different combinations of them. Data frame with  new product features received by text parsing of product name variable NombreProducto from file producto_tabla.csv was merged to main data frame via key Producto_ID. As a result of parsing of product names we got a lot of new product features which denote e.g. short names of product, weight, brand, etc. We grouped the products by clusters and used the number of clusters as a new feature. 

 We chose the data of previous weeks in respect of training sets weeks for aggregations of target variable Demanda_uni_equil, grouped by factors variables. We calculated mean, min, max, sum of target variable grouped by factors and their combinations. Then these aggregated values was merged into main data sets as the new features using the list of these factors as keys for merging. We also calculated the similar aggregated features grouped by factors for variables Venta_hoy, Venta_uni_hoy, Dev_uni_proxima, Dev_uni_proxima. 

We chose the set of 8-9 weeks for the validation and 6-7 weeks’ data were considered as a training set for validation. It was necessary to build 2-level models and find appropriate weights for models blending. 8-9th  weeks were chosen as a training set for prediction sales of the 10-11th  weeks. So, we generated two types of training the data set: 6-7th  weeks training data set for the prediction of 8-9th  weeks sales and 8-9th  weeks for the prediction of 10-11th weeks sales. Since we used the 1st week lag for the target variable as a feature,  we needed to calculate this feature additionally for predicting 11th week sales. First we predicted the10th week, then using the predicted target variable for the 10th week, we calculated lagged features with the lag for 1 week and used these calculated features for the prediction of 11th week sales. As the first step for validation, we calculated the prediction for the 8th week, then using predicted target values we calculated 1 week lag for 9th week data set, and then we made the prediction for 9th week sales. It was done to build the validation model similar to the prediction of 10-11 weeks sales. For lag values, we investigated two types of cases with lags for 1-3 weeks and lags for 2-3 weeks. First we used up to 5 weeks lags, on the next study we used maximum 3 weeks lags. In the case of 1-3 weeks lags, for the validation on 8-9th weeks sales, we made the prediction in 2 steps. For building a two-level model, we used only the validation of 9th  week sales. Our study shows that using 1-3 weeks lags gives us better scores comparing with 2-3 weeks lags. For the features with 1 week lags we used only the mean value of target variable grouped by the list of factor variables.  For 2-3 weeks lags, we also used averaged lags for such features as Dev_proxima, Dev_uni_proxima, Venta_hoy grouped by the list of different factors. Most of the lagged features which were used for classification have 2 weeks lag. We generated more than 300 features.  All received results for the validation on 9th week sales and for the prediction of 10-11th weeks, were used for building two-level models. Our study also shows that using only 7th  week data for the validation on 8-9th  weeks and 9th week for the prediction of sales on 10-11th weeks gives better scores comparing with two weeks training sets, so we built our next single models based on the sets of 7th week for validation and on the sets of 9th week for the prediction. To speed up our calculation we selected 175 top features using function xgb.importance() of ‘xgboost’ R package and then we worked with this set of features. We  also calculated the predictions for Venta_hoy", "Venta_uni_hoy", "Dev_uni_proxima", "Dev_proxima" variables which were used on the 2nd level model. On the first level each of us used his own classifier options and features sets which gave results with good scores for single models without high correlation that is important for creating 2 level models. 

#### Training Methods
For creating single models of the first level, we used xgboost classifier. The options for XGBoost classifiers can be found in our scripts. On the second level, we used validation results of the first level models as training sets for the prediction of target variable for the test set. We created a linear model and model based on ExtraTreesClassifier. On the third level, the results from the second level models were just weighted average.



Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: