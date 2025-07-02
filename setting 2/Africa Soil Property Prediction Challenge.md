You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Africa_Soil_Property_Prediction_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Advances in rapid, low cost analysis of soil samples using infrared spectroscopy, georeferencing of soil samples, and greater availability of earth remote sensing data provide new opportunities for predicting soil functional properties at unsampled locations. Soil functional properties are those properties related to a soil’s capacity to support essential ecosystem services such as primary productivity, nutrient and water retention, and resistance to soil erosion. Digital mapping of soil functional properties, especially in data sparse regions such as Africa, is important for planning sustainable agricultural intensification and natural resources management.

Diffuse reflectance infrared spectroscopy has shown potential in numerous studies to provide a highly repeatable, rapid and low cost measurement of many soil functional properties. The amount of light absorbed by a soil sample is measured, with minimal sample preparation, at hundreds of specific wavebands across a range of wavelengths to provide an infrared spectrum (Fig. 1). The measurement can be typically performed in about 30 seconds, in contrast to conventional reference tests, which are slow and expensive and use chemicals.
Conventional reference soil tests are calibrated to the infrared spectra on a subset of samples selected to span the diversity in soils in a given target geographical area. The calibration models are then used to predict the soil test values for the whole sample set. The predicted soil test values from georeferenced soil samples can in turn be calibrated to remote sensing covariates, which are recorded for every pixel at a fixed spatial resolution in an area, and the calibration model is then used to predict the soil test values for each pixel. The result is a digital map of the soil properties.
This competition asks you to predict 5 target soil functional properties from diffuse reflectance infrared spectroscopy measurements.
Acknowledgements
This competition is sponsored by the Africa Soil Information Service.

##  Evaluation Metric:
Submissions are scored on MCRMSE (mean columnwise root mean squared error):

$$\textrm{MCRMSE} = \frac{1}{5}\sum_{j=1}^{5}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2},$$
where \\(y\\) and \\(\hat{y}\\) are the actual and predicted values, respectively.

**Submission File**
For each row in the dataset, the submission file should contain an identifier column (PIDN) and 5 prediction columns: Ca, P, pH, SOC, and Sand. PIDN, the sample identifier, should be copied from the first column of test data file. Ca, P, pH, SOC, and Sand are soil properties whose values you must predict.

The file should contain a header and have the following format:
    PIDN,Ca,P,pH,SOC,SandXNhoFZW5,1.3,3.1,2.0,1.5,4.8

##  Dataset Description:
File descriptions

    train.csv - the training set has 1158 rows.
    test.csv - the test set has 728 rows.
    sample_submission.csv - all zeros prediction, serving as a sample submission file in the correct format.

Data fields
SOC, pH, Ca, P, Sand are the five target variables for predictions. The data have been monotonously transformed from the original measurements and thus include negative values. 

    PIDN: unique soil sample identifier
    SOC: Soil organic carbon
    pH: pH values
    Ca: Mehlich-3 extractable Calcium
    P: Mehlich-3 extractable Phosphorus
    Sand: Sand content 
    m7497.96 - m599.76: There are 3,578 mid-infrared absorbance measurements. For example, the "m7497.96" column is the absorbance at wavenumber 7497.96 cm-1. We suggest you to remove spectra CO2 bands which are in the region m2379.76 to m2352.76, but you do not have to.
    Depth: Depth of the soil sample (2 categories: "Topsoil", "Subsoil")

We have also included some potential spatial predictors from remote sensing data sources. Short variable descriptions are provided below and additional descriptions can be found at AfSIS data. The data have been mean centered and scaled.

    BSA: average long-term Black Sky Albedo measurements from MODIS satellite images (BSAN = near-infrared, BSAS = shortwave, BSAV = visible)
    CTI: compound topographic index calculated from Shuttle Radar Topography Mission elevation data
    ELEV: Shuttle Radar Topography Mission elevation data
    EVI: average long-term Enhanced Vegetation Index from MODIS satellite images.
    LST: average long-term Land Surface Temperatures from MODIS satellite images (LSTD = day time temperature, LSTN = night time temperature)
    Ref: average long-term Reflectance measurements from MODIS satellite images (Ref1 = blue, Ref2 = red, Ref3 = near-infrared, Ref7 = mid-infrared)
    Reli: topographic Relief calculated from Shuttle Radar Topography mission elevation data
    TMAP & TMFI: average long-term Tropical Rainfall Monitoring Mission data (TMAP = mean annual precipitation, TMFI = modified Fournier index)

## Dataset folder Location: 
../../kaggle-data/afsis-soil-properties. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
This is the solution of Yasser, top-1 private leaderboard. 

### Preprocessing
Different types of preprocessing were done to transform features into more relevant forms. Some of them
reduce dimensionality of data and some others reduce noises. You can use Scipy for such feature selection.

    1- Savitzky-Golay filter: this filter is used for smoothing the data
    2- Continuum Removal: for normalization and handling outliers
    3- Discrete wavelet transforms: for discrete sampling and data reduction
    4- First Derivatives: in some cases increases prediction quality
    5- Unsupervised Feature Selection: standard deviation was used to select top features for some
    algorithms.
    6- Log transform: "P" target was transformed into log(P+1)

### Modeling algorithms
The solution is an ensemble of multiple methods as follows. Note, you can call Scikit-learn for such algorithms.

    1- Neural Networks: two types of neural network algorithms were used for training:
    Simple layer neural network 
    Monotonic Multilayer Perceptron (MONMLP)
    2- Support Vector Machines(SVM)
    3- Multivariate Regression
    4- Gaussian Process
    
For each target different algorithms were used for training. 

| Target | Model Name | Model Weight | Preprocessing Steps | Regression Algorithm |
|-------|--------|-------|------|------|
|Ca | Ca_SVM1 | 0.100 | Savitzky-Golay filter | SVM, cost=1000 |
|| Ca_SVM2| 0.030 | None | SVM, cost=10000 |
|| Ca_SVM3 | 0.100 | None | SVM, cost=5000 |
|| Ca_SVM4 | 0.010 | STD Feature selection(2000 features); First Derivatives; Haar dwt(3 iterations) | SVM, cost=10000 |
|| Ca_MLP1 | 0.100 | STD Feature selection(2000 features); Haar dwt(4 iterations) | Ensemble of 10 MONMLP models (150 iterations,4 neurons in first layer and 4 neurons in second layer) |
|| Ca_MLP2 | 0.100 | Savitzky-Golay filter; Haar dwt(4 iterations) | MONMLP (150 iterations,4 neurons in first layer and 4 neurons in second layer) | 
|| Ca_MLP3 | 0.150 | Haar dwt(5 iterations) | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) ||
|| Ca_MLP4 | 0.050 | First Derivatives;  Haar dwt(7 iterations)| MONMLP (150 iterations,3 neurons in first layer and 20 neurons in second layer) |
|| Ca_MLP5 | 0.030 | Haar dwt(4 iterations); First Derivatives | Two different MONMLP models based on Depth variable (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Ca_MLP6 | 0.030 | Haar dwt(5 iterations) | Two different MONMLP models based on Depth variable (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Ca_MLP7 | 0.150 | Haar dwt(4 iterations) | MONMLP (150 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Ca_MLP8 | 0.050 | Haar dwt(3 iterations) | MONMLP (150 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Ca_Gauss1 | 0.015 | None | GaussPr (rbf kernel) ||
|| Ca_Gauss2 | 0.045 | Multiple Scatter Correction(2 iterations); First Derivatives; Haar dwt(9 iterations); Partial PCA | GaussPr (rbf kernel) |
|| Ca_Gauss3 | 0.010 | None | GaussPr (poly kernel) |
|| Ca_MVR1 | 0.015 | STD Feature selection(2000 features)| MVR(120 components)|
|| Ca_MVR2 | 0.005 | None MVR(100 components) |
|| Ca_NNET1 | 0.010 | Haar dwt(5 iterations) | NNET(10 neurons,100 iterations) |
|P | P_SVM1 | 0.088 | Continuum Removal | SVM, cost=5000 |
|| P_SVM2 | 0.088 | None | SVM, cost=5000 |
|| P_SVM3 | 0.088 | Haar dwt(1 iteration) | Two different SVM models based on Depth variable, cost=1000 |
|| P_MLP1 | 0.088 | Savitzky-Golay filter; STD Feature selection(3000 features); Haar dwt(4 iterations) | MONMLP (150 iterations,5 neurons in first layer and 0 neurons in second layer) |
|| P_MLP3 | 0.125 | Haar dwt(4 iterations); First Derivatives | Two different MONMLP models based on Depth variable (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| P_MLP4 | 0.063 | Haar dwt(5 iterations) | MONMLP (50 iterations,5 neurons in first layer and 5 neurons in second layer)|
|| P_MLP5 | 0.063 | First Derivatives; Haar dwt(2 iterations); STD Feature selection(450 features) | MONMLP (50 iterations,5 neurons in first layer and 5 neurons in second layer) | 
|| P_MLP6 | 0.063 | STD Feature selection(2500 features); Haar dwt(4 iterations); First Derivatives | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| P_MLP7 | 0.250 | STD Feature selection(2500 features); Haar dwt(4 iterations); First Derivatives | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| P_MVR1 | 0.088 | Haar dwt(4 iterations) | MVR(200 components) |
|pH | pH_SVM1 | 0.116 | None| Two different SVM models based on Depth variable, cost=1000 |
|| pH_SVM2 | 0.116 | None | SVM, cost=5000 |
|| pH_MLP1 | 0.163 | Haar dwt(5 iterations) | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| pH_MLP2 | 0.163 | Haar dwt(4 iterations) | Two different MONMLP models based on Depth variable (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| pH_MLP3 | 0.116 | Haar dwt(4 iterations) | MONMLP (150 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| pH_MLP4 | 0.163 | STD Feature selection(2500 features); Haar dwt(4 iterations); First Derivatives | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| pH_MLP5 | 0.163 | STD Feature selection(2500 features); Haar dwt(4 iterations); First Derivatives | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|SOC | SOC_SVM1 | 0.200 | None | SVM, cost=10000 |
|| SOC_SVM2 | 0.140 | None | SVM, cost=5000 |
|| SOC_MLP1 | 0.100 | Savitzky-Golay filter; STD Feature selection(2500 features); Haar dwt(3 iterations) |  MONMLP (150 iterations,3 neurons in first layer and 3 neurons in second layer) |
|| SOC_MLP2 | 0.100 | Savitzky-Golay filter;  Haar dwt(3 iterations); MONMLP (150 iterations,3 neurons in first layer and 3 neurons in second layer) |
|| SOC_MLP3 | 0.100 | Savitzky-Golay filter; STD Feature selection(2500 features); Haar dwt(4 iterations) | MONMLP (150 iterations,4 neurons in first layer and 4 neurons in second layer) |
|| SOC_MLP4 | 0.200 | First Derivatives; Haar dwt(6 iterations) | MONMLP (100 iterations,4 neurons in first layer and 0 neurons in second layer) |
||SOC_MLP5 | 0.120 | Haar dwt(6 iterations) | MONMLP (50 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| SOC_MLP6 | 0.040 | STD Feature selection(2500 features);  Haar dwt(4 iterations);  First Derivatives | MONMLP (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
| Sand | Sand_SVM1 | 0.127 | Savitzky-Golay filter | SVM, cost=5000 |
|| Sand_SVM2 | 0.038 | None | SVM, cost=10000 |
|| Sand_SVM3 | 0.063 | STD Feature selection(2000 features);  First Derivatives; Haar dwt(3 iterations) | SVM, cost=10000 |
|| Sand_SVM4 | 0.063 | STD Feature selection(1500 features); First Derivatives; Haar dwt(3 iterations) | SVM, cost=10000 |
|| Sand_MLP1 | 0.127 | Haar dwt(4 iterations); Savitzky-Golay filter | MONMLP (150 iterations,4 neurons in first layer and 4 neurons in second layer)| 
|| Sand_MLP2 | 0.127 | Haar dwt(4 iterations); PCA | MONMLP (200 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Sand_MLP3 | 0.013 | Haar dwt(5 iterations) | MONMLP (50 iterations,4 neurons in first layer and 0 neurons in second layer) |
|| Sand_MLP4 | 0.089 | Haar dwt(5 iterations) | MONMLP (50 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Sand_MLP5 | 0.152 | First Derivatives; Haar dwt(2 iterations); STD Feature selection(450 features) | MONMLP (50 iterations,5 neurons in first layer and 5 neurons in second layer) |
||Sand_MLP6 | 0.038 | Haar dwt(4 iterations); First Derivatives | Two different MONMLP models based on Depth variable (100 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Sand_MLP7 | 0.076 | Haar dwt(4 iterations) | MONMLP (150 iterations,5 neurons in first layer and 5 neurons in second layer) |
|| Sand_Gauss1 | 0.019 | None | GaussPr (rbf kernel)|
|| Sand_Gauss2 | 0.013 | Multiple Scatter Correction(2 iterations); First Derivatives; Haar dwt(9 iterations); Partial PCA | GaussPr (rbf kernel) |
||Sand_Gauss3 | 0.025 | None | GaussPr (poly kernel) |
|| Sand_MVR1 | 0.019 | STD Feature selection(2000 features)| MVR(120 components)|
|| Sand_NNET1 | 0.013 | Haar dwt(5 iterations)|  NNET(10 neurons,100 iterations)|

Final predictions for each target were calculated using weighted averages of models as detailed in the above table.
Number of training rows was small in compare to number of features, so overfitting could occur.
For handling overfitting risk, the value of C parameter in SVM was set to a large number to increase regularization. Also combining different models significantly reduced drawbacks of single models.  This method won the competition with MCRMSE score of 0.46892.

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: