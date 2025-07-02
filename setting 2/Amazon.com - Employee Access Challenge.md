You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Amazon.com_-_Employee_Access_Challenge_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
When an employee at any company starts work, they first need to obtain the computer access necessary to fulfill their role. This access may allow an employee to read/manipulate resources through various applications or web portals. It is assumed that employees fulfilling the functions of a given role will access the same or similar resources. It is often the case that employees figure out the access they need as they encounter roadblocks during their daily work (e.g. not able to log into a reporting portal). A knowledgeable supervisor then takes time to manually grant the needed access in order to overcome access obstacles. As employees move throughout a company, this access discovery/recovery cycle wastes a nontrivial amount of time and money.

There is a considerable amount of data regarding an employee’s role within an organization and the resources to which they have access. Given the data related to current employees and their provisioned access, models can be built that automatically determine access privileges as employees enter and leave roles within a company. These auto-access models seek to minimize the human involvement required to grant or revoke employee access.

**Objective**
The objective of this competition is to build a model, learned using historical data, that will determine an employee's access needs, such that manual access transactions (grants and revokes) are minimized as the employee's attributes change over time. The model will take an employee's role information and a resource code and will return whether or not access should be granted.

##  Evaluation Metric:
Submissions are judged on  area under the ROC curve. 
In Matlab (using the stats toolbox):
    [~, ~, ~, auc ] = perfcurve(true_labels, predictions, 1);

In R (using the verification package):
    auc = roc.area(true_labels, predictions)

In python (using the metrics module of scikit-learn):
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions, pos_label=1)
    auc = metrics.auc(fpr,tpr)

**Submission File**
For every line in the test set, submission files should contain two columns: id and ACTION. In the ground truth, ACTION is 1 if the resource should be allowed, 0 if the resource should not. Your predictions do not need to be binary. You may submit probabilities/predictions having any real value. The submission file should have a header.

    id,ACTION
    1,1
    2,0.2
    3,1
    4,0
    5,2
    ...

##  Dataset Description:
The data consists of real historical data collected from 2010 & 2011.  Employees are manually allowed or denied access to resources over time. You must create an algorithm capable of learning from this historical data to predict approval/denial for an unseen set of employees. 

**File Descriptions**
**train.csv** - The training set. Each row has the ACTION (ground truth), RESOURCE, and information about the employee's role at the time of approval

**test.csv** - The test set for which predictions should be made.  Each row asks whether an employee having the listed characteristics should have access to the listed resource.
Column Descriptions



| Column Name |  Description |
| -------| -------|
| ACTION | ACTION is 1 if the resource was approved, 0 if the resource was not |
| RESOURCE | An ID for each resource |
| MGR_ID | The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time | 
| ROLE_ROLLUP_1 | Company role grouping category id 1 (e.g. US Engineering) |
| ROLE_ROLLUP_2 | Company role grouping category id 2 (e.g. US Retail) |
| ROLE_DEPTNAME | Company role department description (e.g. Retail) |
| ROLE_TITLE | Company role business title description (e.g. Senior Engineering Retail Manager) |
| ROLE_FAMILY_DESC | Company role family extended description (e.g. Retail Manager, Software Engineering) |
| ROLE_FAMILY | Company role family description (e.g. Retail Manager) |
| ROLE_CODE | Company role code; this code is unique to each role (e.g. Manager) | 

train.csv - column name: ACTION, RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE
test.csv - column name: id, RESOURCE, MGR_ID, ROLE_ROLLUP_1, ROLE_ROLLUP_2, ROLE_DEPTNAME, ROLE_TITLE, ROLE_FAMILY_DESC, ROLE_FAMILY, ROLE_CODE


## Dataset folder Location: 
../../kaggle-data/amazon-employee-access-challenge. In this folder, there are the following files you can use: train.csv, sampleSubmission.csv, test.csv

## Solution Description:
The following is the solution shared by Paul. His solution and Ben ranked at Number 1 position in the private leaderboard.

The final prediction is a weighted average of my code (2/3) and Ben's (1/3).

**Benjamin Solecki**

Ben's approach is itself a blend of a logistic model and a mixture of tree-based models. He explained his approach for the logistic model in more detail here. As for the tree-based models, it is a combination of Random Forests, GBMs, and Extremely Randomized Trees that are grown using features based on counts and frequencies (e.g. number of resources managed by a given manager, etc.). I'll let him explain his approach further if he wishes to -- we merged comparatively late in the game (2 weeks before the deadline) so I would risk misrepresenting his train of thoughts.


**Paul Duan**

As for mine, it was mainly driven by the fact that the structure of the dataset contained all categorical variables, with a large number of categories and some rare features. This meant that stability was a high priority, ie. models should be relatively robust to changes in the composition of the dataset. I believe this is what in the end helped our score to drop less than the other top competitors from the public leaderboard to the private one.

As such:

- I trusted my CV score above all else, which was obtained by repeating a 80/20 train/test split 10 times. I would then select the models not only based on raw score, but also on the standard deviation between the 10 folds.

I made no attempt at fixing the discrepancy between CV and leaderboard scores that was due to the fact that all categories in the test set appeared in the train set, which was not necessarily the case when doing a CV split. The reasoning being that the best model would need to be resistant to a change in the split.

- I removed role 1 and role 2 from the original columns, as their effect was too strong and seemed to cause too much variance; I suspect that the algorithms weren't too good at dealing with unknown role IDs (they were the ones with the fewest number of categories, so the models tended to trust them too much)

- I spent comparatively very little time on feature selection (which was a big subject in this competition, judging by the forums), as they would be highly dependent on the makeup of the dataset. I did, however, reuse some of Miroslaw's code to generate three different datasets built by using greedy feature selection with different seeds. This was not critical to the raw performance of the blend but did help diversifying it/reducing the variance.

- I considered feature extraction to be much more important to feature selection

- my classifier consists of a large combination of models (~15 currently) that are each either using a different algorithm or a different feature set. The top 3-5 models are probably enough to reach .92+ on the private leaderboard, but I found adding additional variants to the datasets helped minimize variance.

I then combined them by computing their predictions using cross-validation, and combining them using a second model (stacking). When training the second model, I also added meta-features (nb of time this resource/manager/etc appeared in the training set, etc.), the idea being to try to determine dynamically which model should be trusted the most (some perform better when the manager is unknown, etc.).

 Each dataset consists of what I called base data sets (combination of the original columns) and extracted feature sets.

The extracted feature sets are based on cross-tabulating all categories and looking at the counts manager/resource/etc X had been associated with role/department/etc Y, and so on (in feature_extraction.py, this is what I called the pivot tables; the features sets are lists of lambda functions that extract the relevant cell in the table).

Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: