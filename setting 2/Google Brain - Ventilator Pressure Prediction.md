You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Google_Brain_-_Ventilator_Pressure_Prediction_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
What do doctors do when a patient has trouble breathing? They use a ventilator to pump oxygen into a sedated patient's lungs via a tube in the windpipe. But mechanical ventilation is a clinician-intensive procedure, a limitation that was prominently on display during the early days of the COVID-19 pandemic. At the same time, developing new methods for controlling mechanical ventilators is prohibitively expensive, even before reaching clinical trials. High-quality simulators could reduce this barrier. 

Current simulators are trained as an ensemble, where each model simulates a single lung setting. However, lungs and their attributes form a continuous space, so a parametric approach must be explored that would consider the differences in patient lungs. 

Partnering with Princeton University, the team at Google Brain aims to grow the community around machine learning for mechanical ventilation control. They believe that neural networks and deep learning can better generalize across lungs with varying characteristics than the current industry standard of PID controllers.  

In this competition, you’ll simulate a ventilator connected to a sedated patient's lung. The best submissions will take lung attributes compliance and resistance into account.

If successful, you'll help overcome the cost barrier of developing new methods for controlling mechanical ventilators. This will pave the way for algorithms that adapt to patients and reduce the burden on clinicians during these novel times and beyond. As a result, ventilator treatments may become more widely available to help patients breathe.
Photo by Nino Liverani on Unsplash

##  Evaluation Metric:
The competition will be scored as the mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored. The score is given by:

|X-Y|

where X is the vector of predicted pressure and Y is the vector of actual pressures across all breaths in the test set.

Submission File

For each id in the test set, you must predict a value for the pressure variable. The file should contain a header and have the following format:

    id,pressure
    1,20
    2,23
    3,24
    etc.


##  Dataset Description:
The ventilator data used in this competition was produced using a modified open-source ventilator connected to an artificial bellows test lung via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

In this competition, participants are given numerous time series of breaths and will learn to predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

Files

    train.csv - the training set
    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format

Columns

    id - globally-unique time step identifier across an entire file
    breath_id - globally-unique time step for breaths
    R - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change R by changing the diameter of the straw, with higher R being harder to blow.
    C - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change C by changing the thickness of the balloon’s latex, with higher C having thinner latex and easier to blow.
    time_step - the actual time stamp.
    u_in - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
    u_out - the control input for the exploratory solenoid valve. Either 0 or 1.
    pressure - the airway pressure measured in the respiratory circuit, measured in cmH2O.

train.csv - column name: id, breath_id, R, C, time_step, u_in, u_out, pressure
test.csv - column name: id, breath_id, R, C, time_step, u_in, u_out


## Dataset folder Location: 
../../kaggle-data/ventilator-pressure-prediction. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv

## Solution Description:
### Summary
Validation strategy

    Stratified k-fold using type_rc as y

Model: We used combination of Conv1d and LSTM

    Architecture: Conv1d + Stacked LSTM
    Optimizer: AdamW
    Scheduler: Cosine

Features: We used 8 features

    Original: u_in, u_out, R(one-hot), C(one-hot)
    Engineered: u_in_min, u_in_diff, inhaled_air, time_diff

Data augmentation: We used three type of augmentations

    Masking: Randomly mask the R or C
    Shuffling: Randomly shuffle our sequences
    Mixup: Select two sequences and mix up them

Loss: We used multi-task MAE loss

    MAE of pressure
    Difference of pressure

Ensemble: We used iterative pseudo labeling and seed ensemble

Seed ensemble using median
1st Pseudo labeling
2nd Pseudo labeling

### Main
I will describe our solution in the following six structures: Validation strategy, Model, Features, Data augmentation, Loss, and Ensemble.

#### Validation strategy
What is the most important thing in a competition? IMHO, It`s to avoid shake-up when the competition is finalized and thus a well-designed validation set is needed to avoid shake-up. Although the validation set was not important to this competition, this is why I first focus on validation strategy when competitions start.

We knew that the types of R and C were very important in this competition through this discussion (thanks to @cdeotte), and we made a validation set with the same ratio of R and C for every fold.

Detailed pseudo-code is below.

```python
cols = ['breath_id', 'R', 'C']

meta_df = train_df.groupby('breath_id')[cols].head(1)
meta_df['RC'] = meta_df['R'] + '_' + meta_df['C']
meta_df['fold'] = -1

kf = StratifiedKFold(12, random_state=seed, shuffle=True)

for fold, (_, val_idx) in enumerate(kf.split(meta_df, meta_df['RC'])):
    meta_df.loc[val_idx, 'fold'] = int(fold)  
```
In this way, we were able to create a validation set that was highly correlated with test data, but I thought that it would be possible to create a good validation set even if we just used normal KFold.

#### Model
After setting the validation set, we focused on boosting the performance of the model using only the original features.

Since Limerobot, the magician of transformers, was on our team, some kagglers expected that our team would use transformers, but we just used simple LSTM models (thanks to @tenffe from this notebook).

Detailed model architecture is below.

1. Input Layer
	•	Input Shape:
	•	The input consists of sequences with shape (BS, SEQ, N), where:
	•	BS is the batch size.
	•	SEQ is the sequence length.
	•	N is the feature size of each element in the sequence.

2. Convolutional Layers (Conv1d)
	•	Three 1D Convolutional layers are applied to the input sequences to extract features using different kernel sizes:
	1.	Conv1d(N, N, ks=2):
	•	Kernel size = 2.
	•	Preserves the feature dimension (N).
	•	Outputs a tensor of shape (BS, SEQ, N).
	2.	Conv1d(N, N, ks=3):
	•	Kernel size = 3.
	•	Preserves the feature dimension (N).
	•	Outputs a tensor of shape (BS, SEQ, N).
	3.	Conv1d(N, N, ks=4):
	•	Kernel size = 4.
	•	Preserves the feature dimension (N).
	•	Outputs a tensor of shape (BS, SEQ, N).

3. Concatenation
	•	The outputs of the three convolutional layers are concatenated along the feature dimension.
	•	Resulting Shape: (BS, SEQ, 4N):
	•	The feature size is now 4 times the original (4N), due to the concatenation.

4. Stacked Bidirectional LSTMs
	•	The concatenated sequences are passed through four bidirectional LSTM layers with decreasing hidden state sizes:
	1.	LSTM(4N, 1024):
	•	Input size: 4N.
	•	Hidden state size: 1024.
	•	Dropout: 0.1.
	2.	LSTM(2048, 512):
	•	Input size: 2048.
	•	Hidden state size: 512.
	•	Dropout: 0.1.
	3.	LSTM(1024, 256):
	•	Input size: 1024.
	•	Hidden state size: 256.
	•	Dropout: 0.1.
	4.	LSTM(512, 128):
	•	Input size: 512.
	•	Hidden state size: 128.
	•	Dropout: 0.1.
	•	The final LSTM layer outputs sequences with shape (BS, SEQ, 256).

5. Fully Connected (FC) Layer
	•	A dense layer maps the LSTM output (256) to the target output dimensions:
	•	Input Size: 256.
	•	Output Size: 1.
	•	Produces two outputs: Pressure and Pressure Difference.

6. Outputs
	•	The final outputs are two sequences of shape (BS, SEQ):
	1.	Pressure: Predicted pressure values for each time step in the sequence.
	2.	Pressure Diff: Predicted pressure difference values for each time step in the sequence.

Our trial-and-error with models related to the pytorch and tensorflow libraries. Initially, most high-score public notebooks were implemented in tensorflow, and we tried to port it to pytorch version. However, there was a low pyotrch score because of a difference between tensorflow and pytorch in the model initialization method (thanks to @junkoda for sharing this notebook). Model initialization methods like tensorflow allowed us to develop models in pytorch.

Detailed code is below (from JUN KODA`s notebook).

```python
for name, p in self.named_parameters():
    if 'lstm' in name:
        if 'weight_ih' in name:
            nn.init.xavier_normal_(p.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(p.data)
        elif 'bias_ih' in name:
            p.data.fill_(0)
            # Set forget-gate bias to 1
            n = p.size(0)
            p.data[(n // 4):(n // 2)].fill_(1)
        elif 'bias_hh' in name:
            p.data.fill_(0)

    if 'conv' in name:
        if 'weight' in name:
            nn.init.xavier_normal_(p.data)
        elif 'bias' in name:
            p.data.fill_(0)

    elif 'reg' in name:
        if 'weight' in name:
            nn.init.xavier_normal_(p.data)
        elif 'bias' in name:
            p.data.fill_(0)
```
#### Feature
After designing the model, we focused on feature engineering. Features can be divided into original features and engineered features. For the original features, all features except R and C were used as they are, and one-hot encoded R and C were used to show the type information of R and C to the model.

For engineered features, we did a lot of EDA and tried to put features the model could not see. In this context, we have created the following features: u_in_min, u_in_diff, inhaled_air, and time_diff.

Details are below.

```python
df['step'] = list(range(80))*df['breath_id'].nunique()

df['u_in_min'] = df['breath_id'].map(df.groupby('breath_id')['u_in'].min())

df['u_in_diff'] = df['u_in'].diff()
df.loc[df['step']<1, 'u_in_diff'] = 0

df['time_diff'] = df['time_step'].diff()
df.loc[df['step']<1, 'time_diff'] = 0

df['inhaled_air'] = df['time_diff']*df['u_in']
```

#### Data augmentation
After feature engineering, we focused on data augmentation. As a result, data augmentation led us to score high. We used three type of data augmentation as following: Masking, Shuffling, and Mixup.

##### Masking augmentation
In the case of masking, it is a little bit similar to Cutout, but masking is randomly performed on the R and C only. In other words, numeric information such as u_in is left as it is, and only information about the type R and C is intentionally masked. In the case of u_in, it was seen that the correlation with pressure was directly high from EDA, and by intentionally erasing the type information, the model was trained to understand the general semantic of u_in.

Pseudo code is below.
```python
one_sample = get_sample(idx)

# Type masking
if random.random() < .2:
    if random.random() < .5:
        # maskring R
        one_sample['R'][:] = 0
    else:
        # masking C
        one_sample['R'][:] = 0
```

##### Shuffling augmentation
Shuffling randomly shuffles units within a specific window in a sequence. For example, suppose we have a sequence like [0,1,2,3,4,5,6,7] and window size is 3. First randomly shuffle between [0,1,2]; next shuffle [3,4,5] and so on.

Details are below.
```python
one_sample = get_sample(idx)

# Shuffling sequence
if random.random() < .2:
    ws = np.random.choice([2, 4, 5])
    num = max_seq_len // ws

    idx = np.arange(0, max_seq_len)
    for i in range(num):
        np.random.shuffle(idx[i * ws:(i + 1) * ws])
        one_sample = one_sample[idx]
```
##### Mixup augmentation
Finally, let`s talk about the Mixup method, which gave us a huge boost in cv and lb scores. Mixup is similar to the process of creating of synthesis sequence, in which two sequences are randomly selected and mixed. For example, given two sequences like [1,2,3,4,5] and [5,4,3,2,1], mix them in half to produce the sequence [3,3,3,3,3].

At this time, the point we were discussed about was how to mix-up R and C type information because of the importance of type information. In fact, before this augmentation, we were treating R and C as categorical features and embedding them via nn.Embedding. But we were able to mix these R and C features by changing the embeddings as one-hot. If there is a sequence with C 20 and the other sequence C 50, one-hot C would look like this: [0, 1, 0] and [0, 0, 1]. Then mix these two embeddings like this: [0, 1, 0] * 0.5 + [0, 0, 1] * 0.5. Then we can finally get an embedding like this: [0, 0.5, 0.5].

This augmentation improved the single model score on the public leaderboard from 0.117 to 0.1004.

The code is below.
```python
one_sample = get_sample(idx)

# Mixup two sequences
if random.random() < 0.4:
    # Get new index
    idx = np.random.randint(len(indices_by_breath_id))
    sampled_sample = get_sample(idx)
    mix_cols = ['R', 'C', 'numeric_features', 'target']
    k = 0.5
    for col in mix_cols:
        one_sample[col] = (one_sample[col]*k + (1-k)*sampled_sample[col])
```

#### Loss
In fact, loss design and augmentation were done at the same time, but for convenience, the augmentation was explained first.

When we analyzed the relationship between u_in and pressure, it was confirmed that the value of u_in of a specific step affects the pressure of a next step. In other words, the flow of air when step is 0 affects pressure of step 1 or pressure of step 2. Therefore, when air enters at a certain point in time, it is delayed and reflected in the later pressure. Thus, we let our model predict the difference of pressure as well.

Using this loss, we were able to improve the cv score from 0.14x to 0.127x, and as a result of median ensemble, we were able to rise to 0.11x in the public leaderboard. IMHO, the huge gap in public leaderboard may be due to this loss.

code
```python
df['step'] = list(range(80))*df['breath_id'].nunique()

df['pressure_diff1'] = df['pressure'].diff(periods=1)
df['pressure_diff2'] = df['pressure'].diff(periods=2)
df['pressure_diff3'] = df['pressure'].diff(periods=3)
df['pressure_diff4'] = df['pressure'].diff(periods=4)
df.loc[df['step']<1, 'pressure_diff1'] = 0
df.loc[df['step']<2, 'pressure_diff2'] = 0
df.loc[df['step']<3, 'pressure_diff3'] = 0
df.loc[df['step']<4, 'pressure_diff4'] = 0    
```

#### Ensemble
Through the above processes, we were able to record 0.1004 as a single model on the public leaderboard, and achieved a score of 0.0975 when applying median ensemble and round post-processing (thanks to @cdeotte). After that, we scored 0.963 by doing seed ensemble (seed42 and seed7).

Then we created another two model with different seeds using pseudo labels generated from 0.963 submission, and we could get to 0.948 to public leaderboard.

Finally, when we pseudo-labeled the submission with 0.948 files one more and training our model, then we were able to finally achieve 0.0942 for the public leaderboard and 0.0970 for the private leaderboard.




Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: