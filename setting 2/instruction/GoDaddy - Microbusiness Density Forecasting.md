You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named GoDaddy_-_Microbusiness_Density_Forecasting_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description

#### Goal of the Competition

The goal of this competition is to predict monthly microbusiness density in a given area. You will develop an accurate model trained on U.S. county-level data.

Your work will help policymakers gain visibility into microbusinesses, a growing trend of very small entities. Additional information will enable new policies and programs to improve the success and impact of these smallest of businesses.

#### Context

American policy leaders strive to develop economies that are more inclusive and resilient to downturns. They're also aware that with advances in technology, entrepreneurship has never been more accessible than it is today. Whether to create a more appropriate work/life balance, to follow a passion, or due to loss of employment, studies have demonstrated that Americans increasingly choose to create businesses of their own to meet their financial goals. The challenge is that these "microbusinesses" are often too small or too new to show up in traditional economic data sources, making it nearly impossible for policymakers to study them. But data science could help fill in the gaps and provide insights into the factors associated these businesses.

Over the past few years the Venture Forward team at GoDaddy has worked hard to produce data assets about the tens of millions of microbusinesses in the United States. Microbusinesses are generally defined as businesses with an online presence and ten or fewer employees. GoDaddy has visibility into more than 20 million of them, owned by more than 10 million entrepreneurs. We've surveyed this universe of microbusiness owners for several years and have collected a great deal of information on them that you can access via our survey data here.

Current models leverage available internal and census data, use econometric approaches, and focus on understanding primary determinants. While these methods are adequate, there's potential to include additional data and using more advanced approaches to improve predictions and to better inform decision-making.

Competition host GoDaddy is the world’s largest services platform for entrepreneurs around the globe. They're on a mission to empower their worldwide community of 20+ million customers—and entrepreneurs everywhere—by giving them all the help and tools they need to grow online.

Your work will help better inform policymakers as they strive to make the world a better place for microbusiness entrepreneurs. This will have a real and substantial impact on communities across the country and will help our broader economy adapt to a constantly evolving world.

##  Evaluation Metric:
Submissions are evaluated on SMAPE between forecasts and actual values. We define SMAPE = 0 when the actual and predicted values are both 0.

Submission File

For each row_id you must predict the microbusiness_density. The file should contain a header and have the following format:

    row_id,microbusiness_density
    1001_2022-11-01,1.2
    1002_2022-11-01,2.3
    1003_2022-11-01,3.4
    etc.

The submission file will remain unchanged throughout the competition. However, the actively scored dates will be updated as new data becomes available. During the active phase of the competition only the most recent month of data will be used for the public leaderboard.

##  Dataset Description:
Your challenge in this competition is to forecast microbusiness activity across the United States, as measured by the density of microbusinesses in US counties. Microbusinesses are often too small or too new to show up in traditional economic data sources, but microbusiness activity may be correlated with other economic indicators of general interest.

As historic economic data are widely available, this is a forecasting competition. The forecasting phase public leaderboard and final private leaderboard will be determined using data gathered after the submission period closes. You will make static forecasts that can only incorporate information available before the end of the submission period. This means that while we will rescore submissions during the forecasting period we will not rerun any notebooks. 

#### Files
A great deal of data is publicly available about counties and we have not attempted to gather it all here. You are strongly encouraged to use external data sources for features.

train.csv

    row_id - An ID code for the row.
    cfips - A unique identifier for each county using the Federal Information Processing System. The first two digits correspond to the state FIPS code, while the following 3 represent the county.
    county_name - The written name of the county.
    state_name - The name of the state.
    first_day_of_month - The date of the first day of the month.
    microbusiness_density - Microbusinesses per 100 people over the age of 18 in the given county. This is the target variable. The population figures used to calculate the density are on a two-year lag due to the pace of update provided by the U.S. Census Bureau, which provides the underlying population data annually. 2021 density figures are calculated using 2019 population figures, etc.
    active - The raw count of microbusinesses in the county. Not provided for the test set.

sample_submission.csv A valid sample submission. This file will remain unchanged throughout the competition.

    row_id - An ID code for the row.
    microbusiness_density - The target variable.

test.csv Metadata for the submission rows. This file will remain unchanged throughout the competition.

    row_id - An ID code for the row.
    cfips - A unique identifier for each county using the Federal Information Processing System. The first two digits correspond to the state FIPS code, while the following 3 represent the county.
    first_day_of_month - The date of the first day of the month.

revealed_test.csv During the submission period, only the most recent month of data will be used for the public leaderboard. Any test set data older than that will be published in revealed_test.csv, closely following the usual data release cycle for the microbusiness report. We expect to publish one copy of revealed_test.csv in mid February. This file's schema will match train.csv.

census_starter.csv Examples of useful columns from the Census Bureau's American Community Survey (ACS) at data.census.gov. The percentage fields were derived from the raw counts provided by the ACS. All fields have a two year lag to match what information was avaiable at the time a given microbusiness data update was published.

    pct_bb_[year] - The percentage of households in the county with access to broadband of any type. Derived from ACS table B28002: PRESENCE AND TYPES OF INTERNET SUBSCRIPTIONS IN HOUSEHOLD.
    cfips - The CFIPS code.
    pct_college_[year] - The percent of the population in the county over age 25 with a 4-year college degree. Derived from ACS table S1501: EDUCATIONAL ATTAINMENT.
    pct_foreign_born_[year] - The percent of the population in the county born outside of the United States. Derived from ACS table DP02: SELECTED SOCIAL CHARACTERISTICS IN THE UNITED STATES.
    pct_it_workers_[year] - The percent of the workforce in the county employed in information related industries. Derived from ACS table S2405: INDUSTRY BY OCCUPATION FOR THE CIVILIAN EMPLOYED POPULATION 16 YEARS AND OVER.
    median_hh_inc_[year] - The median household income in the county. Derived from ACS table S1901: INCOME IN THE PAST 12 MONTHS (IN 2021 INFLATION-ADJUSTED DOLLARS).

train.csv - column name: row_id, cfips, county, state, first_day_of_month, microbusiness_density, active
revealed_test.csv - column name: row_id, cfips, county, state, first_day_of_month, microbusiness_density, active
census_starter.csv - column name: pct_bb_2017, pct_bb_2018, pct_bb_2019, pct_bb_2020, pct_bb_2021, cfips, pct_college_2017, pct_college_2018, pct_college_2019, pct_college_2020, pct_college_2021, pct_foreign_born_2017, pct_foreign_born_2018, pct_foreign_born_2019, pct_foreign_born_2020, pct_foreign_born_2021, pct_it_workers_2017, pct_it_workers_2018, pct_it_workers_2019, pct_it_workers_2020, pct_it_workers_2021, median_hh_inc_2017, median_hh_inc_2018, median_hh_inc_2019, median_hh_inc_2020, median_hh_inc_2021
test.csv - column name: row_id, cfips, first_day_of_month


## Dataset folder Location: 
../../kaggle-data/godaddy-microbusiness-density-forecasting. In this folder, there are the following files you can use: train.csv, revealed_test.csv, census_starter.csv, test.csv, sample_submission.csv

## Solution Description:
Interesting problem, and it was also interesting to see the solutions came up with. I'm pretty new to these competitions. The biggest uncertainty to me is how much probing of the hidden data is allowed, especially in a case like this where it matters a lot for time series prediction.

I also though SMAPE was a weird choice of metric that didn't make sense in a business sense. A change in a small county would have a large influence in SMAPE, where if you are trying to capture revenue streams, wouldn't you care a lot more about a change in the number of active forecasts in Los Angeles? I can only imagine this would be useful for allocation of advertising dollars.

### The biggest issue - Data quality
As has been pointed out by many people, SMAPE is a relative metric. With the kind of forecast values we are looking at (~1.2-1.5 for 1 month, ~3.2 for months 3-5 together), we can look at the contribution of an individual CFIPS.

$$\Delta_{SMAPE,i} = \frac{200}{n_{counties}} \frac{|F_i - A_i|}{F_i+A_i}$$

Which, in the case of something going from zero to non-zero can be ~0.0638 per month difference. At the same time, a prediction for most of these models seems to be on the order of 0.5% change per month, which is far below the smallest change in active entries for many 25% of the counties.

Second, there are many CFIPS where the data is terrible. The hosts acknowledged that they had a methodology change in Jan 2021, leading to a number of jumps, and I suspect there was another change after the first month.

The one which always bothered me was CFIPS 56033, Sheridan Co, WY, where there are 2.36 microbusinesses/working age person. I can only guess there is some bug where if there is a misclassification they dump it in that CFIPS. I also wondered if there was some fraud happening during COVID times as people chased PPP loans given the large jumps. However, many of these issues would eventually revert.

So the question I think that make or breaks this, if we identify a large jump, will it revert? In this I believe we are helped by the gap between December and the first forecast month, March. I also hope GoDaddy gets better in their data collection procedures, which would help. In then end, if they want to use this in business, it would probably not be that useful to have a large machine learning error correction model for data collection issues.

My solution is a mix of public leaderboard probing for individual CFIPS changes, reversion for outlier CFIPS, and a forecast of the smooth changes.

### Continuous Model
After seeing GiBa's @titericz notebook, his data cleaning method reminded me of something used in futures algorithmic trading, called a continuous contract. Basically, you need to take a discontinuous series of prices and make it smooth. So for what I will call the continuous forecast, I used a data cleaning method where I looked for large jumps in the number of active entries, then did a shift to smooth them out. So the month where there was a large active jump became zero active jump. This smoothing method gave the best CV score of the number that I tried.

I then setup the CV environment. For the model used XGBoost, added extra indicators, and prevented peeking in the future:

    lagged density changes and lagged active values
    pct_bb, pct_college, pct_it, pct_foreign_born, median_hh_inc of the last year (Many public books used an implicit forward bias in their features by looking at 2021 census data when training on 2019 data)
    Labor Force and Labor force participation for the county
    10 year average pct population change for the county
    latitude and longitude
    engagement, participation, and MAI_composite from the Godaddy website.
    The difference between the microbusiness density and the average of the neighboring counties, weighted by population.

I found a population cut of about 5,000 was helpful. Rounding the model to an integer number of active helped a bit. A lot of these external indicators proved more helpful for the longer term forecasts, where there is some reversion to an average.

Other notes:

    I actually found training on the 1 month forward change then feeding that prediction back into the model and recalculating all indicators gave the best CV, as opposed to directly forecasting the 3 month ahead forecast. It turned out to be less biased.
    The scale variable used seemed to work in forecasting the public leaderboard, but did terribly in my CV environment.
    Again, noting that the last value is pretty good and I was using a roll forward model, I used the tuned public models to patch in a January forecast as an estimate for the ground truth, then rolled forward the XGBoost model from that.


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: