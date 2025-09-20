You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named How_Much_Did_It_Rain?_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
For agriculture, it is extremely important to know how much it rained on a particular field. However, rainfall is variable in space and time and it is impossible to have rain gauges everywhere. Therefore, remote sensing instruments such as radar are used to provide wide spatial coverage. Rainfall estimates drawn from remotely sensed observations will never exactly match the measurements that are carried out using rain gauges, due to the inherent characteristics of both sensors. Currently, radar observations are "corrected" using nearby gauges and a single estimate of rainfall is provided to users who need to know how much it rained. This competition will explore how to address this problem in a probabilistic manner.  Knowing the full probabilistic spread of rainfall amounts can be very useful to drive hydrological and agronomic models -- much more than a single estimate of rainfall.

Image courtesy of NOAA

Unlike a conventional Doppler radar, a polarimetric radar transmits radio wave pulses that have both horizontal and vertical orientations. Because rain drops become flatter as they increase in size and because ice crystals tend to be elongated vertically, whereas liquid droplets tend to be flattened, it is possible to infer the size of rain drops and the type of hydrometeor from the differential reflectivity of the two orientations.

In this competition, you are given polarimetric radar values and derived quantities at a location over the period of one hour. You will need to produce a probabilistic distribution of the hourly rain gauge total. More details are on the data page.

This competition is sponsored by the Artificial Intelligence Committee of the American Meteorological Society. The Climate Corporation has kindly agreed to sponsor the prizes.

##  Evaluation Metric:
The winning entry will be the one that minimizes the Continuous Ranked Probability Score:
$$C = \frac{1}{70N} \sum_{N} \sum_{n=0}^{69} (P(y \le n) -H(n -z))^2$$
over the testing dataset (of size N) where z is the actually recorded gauge value (in mm) and H(x) is the Heavyside step function i.e. H(x) = 1 for
$$x \ge 0$$
and zero otherwise. The entry will be discarded if any of the answers has 
$$P(y \le k) > P(y \le k+1)$$
for any k, i.e., the CDF has to be non-decreasing.

Submission Instructions

The submission file specifies for each location (with Id), the predicted cumulative probabilities between 0 and 69 mm (both inclusive). There are 70 columns of probabilities for each Id. 

    Id,Predicted0,Predicted1,Predicted2,Predicted3,...,Predicted69
    1,0.493069074336,0.725572625942,0.87785520942,0.951305748155,...,1.0
    2,0.5,0.73105857863,0.880797077978,0.952574126822,...,1.0
    ...
    etc

##  Dataset Description:
#### File descriptions

    train_2013.csv.zip - the training set
    test_2014.csv.zip - the test set
    sampleSubmission.csv - a sample submission file in the correct format
    sample_solution.py -- a Python program that is capable of taking test_2014.csv and producing sampleSubmission.csv

#### Predicting probabilities
In this competition, you are given polarimetric radar values and derived quantities at a location over the period of one hour. You will need to produce a probabilistic distribution of the hourly rain gauge total, i.e., produce 
$$P(y \le Y)$$
where y is the rain accumulation and Y lies between 0 and 69 mm (both inclusive) in increments of 1 mm. For every row in the dataset, submission files should contain 71 columns: Id and 70 numbers. 

#### Understanding the data
The training data consists of NEXRAD and MADIS data collected the first 8 days of Apr to Nov 2013 over midwestern corn-growing states. Time and location information have been censored, and the data have been shuffled so that they are not ordered by time or place. The test data consists of data from the same radars and gauges over the same months but in 2014. Please see this page to understand more about polarimetric radar measurements.

#### Data columns
The columns in the datasets are:

    TimeToEnd:  How many minutes before the end of the hour was this radar observation?
    DistanceToRadar:  Distance between radar and gauge.  This value is scaled and rounded to prevent reverse engineering gauge location
    Composite:  Maximum reflectivity in vertical volume above gauge
    HybridScan: Reflectivity in elevation scan closest to ground
    HydrometeorType:  One of nine categories in NSSL HCA. See presentation for details.
    Kdp:  Differential phase
    RR1:  Rain rate from HCA-based algorithm
    RR2:  Rain rate from Zdr-based algorithm
    RR3:  Rain rate from Kdp-based algorithm
    RadarQualityIndex:  A value from 0 (bad data) to 1 (good data)
    Reflectivity:  In dBZ
    ReflectivityQC:  Quality-controlled reflectivity
    RhoHV:  Correlation coefficient
    Velocity:  (aliased) Doppler velocity
    Zdr:  Differential reflectivity in dB
    LogWaterVolume:  How much of radar pixel is filled with water droplets?
    MassWeightedMean:  Mean drop size in mm
    MassWeightedSD:  Standard deviation of drop size
    Expected: the actual amount of rain reported by the rain gauge for that hour.

Composite columns

When the reflectivity composite is provided, you get multiple values in one column (note that they are not comma-separated):
13.5 12.5 21.5 17.5 13.0 -99900.0 13.0 -99900.0 12.0 18.5 24.5 -99900.0 18.0 -99900.0 17.0 18.0 15.5 15.0 10.5 14.0 14.0,-99900.0 -99900.0 12.0 -99900.0 15.5 -99900.0 11.5 -99900.0 -99900.0 -99900.0 15.0 17.5 15.5 -99900.0 13.5 -99900.0 9.5 9.0 16.5 17.5 14.0

This is because there are multiple radar observations (one per radar volume scan) within an hour and because there are two radars that observe the atmosphere above this rain gauge.  

To determine there are two radars, look at the "TimeToEnd" column and notice that there are two sequences that count down to zero:

58.0 55.0 52.0 49.0 41.0 39.0 30.0 25.0 17.0 14.0 11.0 9.0 60.0 58.0 55.0 52.0 50.0 48.0 46.0 43.0 26.0

You could also use the "DistanceToRadar" column, but theoretically, there could be a rain gauge that is equidistant from two radars.  

Each column in a row will have the same number of values that is reflected by the "TimeToEnd" column.

#### Data encoding

Hydrometeor types:

    0-no echo; 
    1-moderate rain; 
    2-moderate rain; 
    3-heavy rain; 
    4-rain/hail; 
    5-big drops;
    6-AP; 
    7-Birds; 
    8-unknown; 
    9-no echo; 
    10-dry snow; 
    11-wet snow; 
    12-ice crystals; 
    13-graupel;
    14-graupel
There are five types of "missing data" codes in the dataset:

    -99000: echo below signal-to-noise threshold of radar.  In other words, the true value could be anywhere between -14 and -inf, but we don't know.
    -99901: range folded data
    -99903: data not collected such as due to beam blockage or beyond data range
    nan: derived quantity could not be computed because some input was one of the above codes9
    99.0: RadarQualityIndex could not be computed because pixel was at edge of echo

#### Data anonymization
We have anonymized and shuffled the data to remove time and location information. However, just to cover our bases, we will state that you are not allowed to infer the rain gauge corresponding to the input data and use the actual reported value from that rain gauge -- any entry that uses rain gauge data beyond that supplied in the problem will be disqualified.

train_2013.csv - column name: Id, TimeToEnd, DistanceToRadar, Composite, HybridScan, HydrometeorType, Kdp, RR1, RR2, RR3, RadarQualityIndex, Reflectivity, ReflectivityQC, RhoHV, Velocity, Zdr, LogWaterVolume, MassWeightedMean, MassWeightedSD, Expected
test_2014.csv - column name: Id, TimeToEnd, DistanceToRadar, Composite, HybridScan, HydrometeorType, Kdp, RR1, RR2, RR3, RadarQualityIndex, Reflectivity, ReflectivityQC, RhoHV, Velocity, Zdr, LogWaterVolume, MassWeightedMean, MassWeightedSD


## Dataset folder Location: 
../../kaggle-data/how-much-did-it-rain. In this folder, there are the following files you can use: sampleSubmission.csv, train_2013.csv, test_2014.csv

## Solution Description:

### Summery
This is a description of my solution for the competition How Much Did it Rain?. I used multi-class classification with soft labels to estimate rain amount. The dataset was split based on the number of radar scans to allow for better feature extraction and ameliorate some of the issues surrounding class imbalance. My final model, after debugging and simplification, scored 0.007485 on the private leaderboard and 0.00751 on the public leaderboard.

### Feature Extraction
Each variable in the original dataset was expanded so that each scan had it's own column. The number of error codes were counted(-99903, -99900 etc) for each row, as well as the total number of non-error code features. The error codes were then replaced by np.nan and various descriptive statistics were calculate for each row(ignoring the nan values). For HydrometeorType the number of each code for each row is counted. Using these methods created 191 features which were used for all subsets of the data. 

In addition for the subsets with 8-17 radar scans and for greater then 18 radar scans, TimeToEnd was sorted and used to break the hour into 10 and 20 equal length segments respectively. In each segment the mean was calculated and used as a feature. These features increased the score for the dataset with more then 17 scans by a large amount but did little for subsets with less scans.

### Feature Selection
Though feature selection could remove approximately 50% of the features without worsening the score I did not remove many features. For the subset with only 1 radar scan all columns with identical values were dropped. Additionally as DistanceToRadar, and the count of -99903 and -99900 error codes for HybridScan dramatically worsened the score when the labels above 70mm were included I dropped them. After removal of large rain amount samples these features only slightly decreased the score and but were still left out.

### Modeling Approach
My overall modeling approach was treat the predictions as a linear combination of cdfs that I estimated from the training set. Each component cdf was weighted by the class probability for the cdf's associated label, as given by the classification algorithm. 

The best prediction for a sample with the true rain amount of 0mm is a row of all ones, and for 1mm true rain amount the best prediction is all ones except for a single zero at the 0mm position. Using this it seemed reasonable that if the classifier predicted p(rain=0mm)=0.5 and p(rain=1mm)=0.5 then the best cdf would be the average of all ones, and all ones except at zero([0.5,1,1,...]). 

Based on this intuition I decided to model the final distribution as a linear combination of step functions and empirical distributions from the training set. 

There was a decent amount of class imbalance in the dataset. About 85% of the data had no rain at all, and at rain amounts greater 50mm there were less then ten examples per class. To combat this I aggregated the higher rain amounts into a single class label. I then used the distribution of rain amounts from the training data falling into the aggregate to estimate a cdf. 

While doing feature engineering I decided to split dataset apart based on the number of radar scans. It was not possible to make the same type of features for a sample with 20 radar scans as it was for one with only 1 radar scan. Splitting the dataset decreased training time as I found I could get by with fewer features for a majority of the data. Additionally tackling the problem this was alleviated some of the class imbalance problems for the datasets with more scans. In the dataset with only one scan 95% of the data had 0mm of rain, and there were very few with large amount of rain. This allowed for modeling the dataset with only 3 labels. In contrast for the subset with greater then 17 scans only 48% of the data had 0mm of rain. For this set I was able to use 12 different labels.

Xgboost's Gradient Boosted Decision Trees classifier was used for all supervised learning. Parameters were ballpark estimates of what might prevent over fitting. Whenever I tuned parameters my leaderboard score would either not change or worsen so I stopped tuning. 

At the end I added some post-processing which improved the score by ~0.00001. For each of the subsets of the data I subtracted a constant value from all the predictions. The amount I subtracted was roughly the same as the proportion of radar scans greater then 70 in the train set. I am not sure whether this was a coincidence or that because large rain amounts increased the score the most subtracting a little from each prediction mitigated their impact.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: