You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Santander_Product_Recommendation_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Ready to make a downpayment on your first house? Or looking to leverage the equity in the home you have? To support needs for a range of financial decisions, Santander Bank offers a lending hand to their customers through personalized product recommendations.

Under their current system, a small number of Santander’s customers receive many recommendations while many others rarely see any resulting in an uneven customer experience. In their second competition, Santander is challenging Kagglers to predict which products their existing customers will use in the next month based on their past behavior and that of similar customers.

With a more effective recommendation system in place, Santander can better meet the individual needs of all customers and ensure their satisfaction no matter where they are in life.

Disclaimer: This data set does not include any real Santander Spain's customer, and thus it is not representative of Spain's customer base.

##  Evaluation Metric:
Submissions are evaluated according to the Mean Average Precision @ 7 (MAP@7):

$$MAP@7 = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{1}{min(m, 7)} \sum_{k=1}^{min(n,7)} P(k)$$

where |U| is the number of rows (users in two time points), P(k) is the precision at cutoff k, n is the number of predicted products, and m is the number of added products for the given user at that time point. If m = 0, the precision is defined to be 0.

Submission File

For every user at each time point, you must predict a space-delimited list of the products they added. The file should contain a header and have the following format:

    ncodpers,added_products
    15889,ind_tjcr_fin_ult1
    15890,ind_tjcr_fin_ult1 ind_recibo_ult1
    15892,ind_nomina_ult1
    15893,
    etc.

##  Dataset Description:
In this competition, you are provided with 1.5 years of customers behavior data from Santander bank to predict what new products customers will purchase. The data starts at 2015-01-28 and has monthly records of products a customer has, such as "credit card", "savings account", etc. You will predict what additional products a customer will get in the last month, 2016-06-28, in addition to what they already have at 2016-05-28. These products are the columns named: ind_(xyz)_ult1, which are the columns #25 - #48 in the training data. You will predict what a customer will buy in addition to what they already had at 2016-05-28. 

The test and train sets are split by time, and public and private leaderboard sets are split randomly.

Please note: This sample does not include any real Santander Spain customers, and thus it is not representative of Spain's customer base. 

#### File descriptions

    train.csv - the training set
    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format

Data fields


Column NameDescription

    fecha_dato The table is partitioned for this column
    ncodpers	Customer code
    ind_empleado Employee index: A active, B ex employed, F filial, N not employee, P pasive
    pais_residencia Customer's Country residence
    sexo Customer's sex
    age Age
    fecha_alta The date in which the customer became as the first holder of a contract in the bank
    ind_nuevo New customer Index. 1 if the customer registered in the last 6 months.
    antiguedad Customer seniority (in months)
    indrel 1 (First/Primary), 99 (Primary customer during the month but not at the end of the month)
    ult_fec_cli_1t Last date as primary customer (if he isn't at the end of the month)
    indrel_1mes Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
    tiprel_1mes Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
    indresi Residence index (S (Yes) or N (No) if the residence country is the same than the bank country)
    indext Foreigner index (S (Yes) or N (No) if the customer's birth country is different than the bank country)
    conyuemp Spouse index. 1 if the customer is spouse of an employee
    canal_entrada channel used by the customer to join
    indfall Deceased index. N/S
    tipodom Addres type. 1, primary address
    cod_prov Province code (customer's address)
    nomprov Province name
    ind_actividad_cliente Activity index (1, active customer; 0, inactive customer)
    renta Gross income of the household
    segmento segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
    ind_ahor_fin_ult1 Saving Account
    ind_aval_fin_ult1 Guarantees
    ind_cco_fin_ult1 Current Accounts
    ind_cder_fin_ult1 Derivada Account
    ind_cno_fin_ult1 Payroll Account
    ind_ctju_fin_ult1 Junior Account
    ind_ctma_fin_ult1 Más particular Account
    ind_ctop_fin_ult1 particular Account
    ind_ctpp_fin_ult1 particular Plus Account
    ind_deco_fin_ult1 Short-term deposits
    ind_deme_fin_ult1 Medium-term deposits
    ind_dela_fin_ult1 Long-term deposits
    ind_ecue_fin_ult1 e-account
    ind_fond_fin_ult1 Funds
    ind_hip_fin_ult1 Mortgage
    ind_plan_fin_ult1 Pensions
    ind_pres_fin_ult1 Loans
    ind_reca_fin_ult1 Taxes
    ind_tjcr_fin_ult1 Credit Card
    ind_valo_fin_ult1 Securities
    ind_viv_fin_ult1 Home Account
    ind_nomina_ult1 Payroll
    ind_nom_pens_ult1 Pensions
    ind_recibo_ult1 Direct Debit

train.csv - column name: fecha_dato, ncodpers, ind_empleado, pais_residencia, sexo, age, fecha_alta, ind_nuevo, antiguedad, indrel, ult_fec_cli_1t, indrel_1mes, tiprel_1mes, indresi, indext, conyuemp, canal_entrada, indfall, tipodom, cod_prov, nomprov, ind_actividad_cliente, renta, segmento, ind_ahor_fin_ult1, ind_aval_fin_ult1, ind_cco_fin_ult1, ind_cder_fin_ult1, ind_cno_fin_ult1, ind_ctju_fin_ult1, ind_ctma_fin_ult1, ind_ctop_fin_ult1, ind_ctpp_fin_ult1, ind_deco_fin_ult1, ind_deme_fin_ult1, ind_dela_fin_ult1, ind_ecue_fin_ult1, ind_fond_fin_ult1, ind_hip_fin_ult1, ind_plan_fin_ult1, ind_pres_fin_ult1, ind_reca_fin_ult1, ind_tjcr_fin_ult1, ind_valo_fin_ult1, ind_viv_fin_ult1, ind_nomina_ult1, ind_nom_pens_ult1, ind_recibo_ult1
test.csv - column name: fecha_dato, ncodpers, ind_empleado, pais_residencia, sexo, age, fecha_alta, ind_nuevo, antiguedad, indrel, ult_fec_cli_1t, indrel_1mes, tiprel_1mes, indresi, indext, conyuemp, canal_entrada, indfall, tipodom, cod_prov, nomprov, ind_actividad_cliente, renta, segmento


## Dataset folder Location: 
../../kaggle-data/santander-product-recommendation. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
This solution is an ensemble of 12 neural nets and 8 GBMs with several hundred features, so I'll try to stick to the highlights.  Please let me know if I inadvertently gloss over anything interesting.

#### Features

For the most part, these are similar to what people mentioned on the forums:  lags of products, time since presence of products, average of products, time since last purchase of products, etc.

Some features which I have not seen mentioned elsewhere were time since change and lags for a few non-product attributes: segmento, ind_actividad_cliente, cod_prov, canal_entrada, indrel_1mes, tiprel_1mes. 

Features which seemed to hurt individual submodels but made their way into the ensemble anyway were historical averages of products segmented by combinations of: canal_entrada, segmento, cod_prov.

#### GBM Models

These submodels are all similar to each other but slightly different from the popular approach on the forum.  They are 17-class multinomial models targeting the 16 most popular product additions.  The remaining 17th class indicates either no additions or an addition of one of the eight remaining products.  

As described elsewhere, multiple product additions are handled by adding duplicate rows with different targets.  These duplicate rows are weighted by the reciprocal of the number of added products.  

Adding the 17th "no addition" class expands the amount of training data substantially.  The largest submodel covers Jun-15 to May-16 and weighs in around 10 million rows.  LightGBM is key to building these models in an expeditious manner.
Differences between submodels are primarily due to training on different time frames and using varying combinations of features.

#### NN Models

Rather than target the addition of a product, these submodels target presence of product in a given month.  They are ambivalent whether the product is new or whether the customer has carried the product all along.

These models are multi-target rather than multinomial.  They target a length 16 vector of the more more popular products and are trained on all customers regardless of whether they added a product.

The structure of all the nets is the same.  They have an input layer, two hidden layers of 512 nodes, and the 16 node output layer.  The largest training set here is also around 10 million rows but Keras made these nets easy to set up and relatively quick to build.

Differences between these models are again on time frames and features as well as multiple runs with different seeds.

#### Post Processing

Where applicable, each submodel is scored once as Jun-16, once as Jun-15, and once as Dec-15.  By "score as" I mean that fecha_dato is incorporated into the models by converting it to numeric, 1 to 18.  Modifying fecha_dato on the test set causes us to "score as" a different month.

Generally speaking, the Jun-16 scores for each submodel are retained with the exception of the reca score, which is replaced by the Jun-15 score, and the cco score, which is replaced by the Dec-15 score.

Submodel scores are set to zero when the customer had a product in the previous month.  The individual product scores are then baselined to level of public leaderboard.  This is accomplished by multiplying by the ratio of the average product score to the value obtained on the public leaderboard by submitting only that product.

The final ensemble is a weighted average of the submodels with weights obtained from leaderboard feedback.

Thanks for freading and thanks to Santander and Kaggle for making this competition happen..

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: