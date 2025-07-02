You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Child_Mind_Institute_-_Detect_Sleep_States_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Your work will improve researchers' ability to analyze accelerometer data for sleep monitoring and enable them to conduct large-scale studies of sleep. Ultimately, the work of this competition could improve awareness and guidance surrounding the importance of sleep. The valuable insights into how environmental factors impact sleep, mood, and behavior can inform the development of personalized interventions and support systems tailored to the unique needs of each child.

#### Goal of the Competition

The goal of this competition is to detect sleep onset and wake. You will develop a model trained on wrist-worn accelerometer data in order to determine a person's sleep state.

Your work could make it possible for researchers to conduct more reliable, larger-scale sleep studies across a range of populations and contexts. The results of such studies could provide even more information about sleep.

The successful outcome of this competition can also have significant implications for children and youth, especially those with mood and behavior difficulties. Sleep is crucial in regulating mood, emotions, and behavior in individuals of all ages, particularly children. By accurately detecting periods of sleep and wakefulness from wrist-worn accelerometer data, researchers can gain a deeper understanding of sleep patterns and better understand disturbances in children.

#### Context

The “Zzzs” you catch each night are crucial for your overall health. Sleep affects everything from your development to cognitive functioning. Even so, research into sleep has proved challenging, due to the lack of naturalistic data capture alongside accurate annotation. If data science could help researchers better analyze wrist-worn accelerometer data for sleep monitoring, sleep experts could more easily conduct large-scale studies of sleep, thus improving the understanding of sleep's importance and function.

Current approaches for annotating sleep data include sleep logs, which are the gold standard for detecting the onset of sleep. However, they are impractical for many participants to use reliably, and fail to capture the nuanced difference between heading to bed and falling asleep (or, conversely, waking up and getting out of bed). Heuristic-based software is another solution that attempts to identify sleep windows, though these rely on human-engineered features of sleep (i.e. arm angle) that vary across individuals and don’t accurately summarize the sleep windows that experts can visually detect from their data. With improved tools to analyze sleep data on a large scale, researchers can explore the relationship between sleep and mood/behavioral difficulties. This knowledge can lead to more targeted interventions and treatment strategies.

Competition host Child Mind Institute (CMI) transforms the lives of children and families struggling with mental health and learning disorders by giving them the help they need. CMI has become the leading independent nonprofit in children’s mental health by providing gold-standard evidence-based care, delivering educational resources to millions of families each year, training educators in underserved communities, and developing tomorrow’s breakthrough treatments.

Established with a foundational grant from the Stavros Niarchos Foundation (SNF), the SNF Global Center for Child and Adolescent Mental Health at the Child Mind Institute works to accelerate global collaboration on under-researched areas of children’s mental health and expand worldwide access to culturally appropriate trainings, resources, and treatment. A major goal of the SNF Global Center is to expand innovations in clinical assessment and intervention, to include building, testing, and deploying new technologies to augment mental health care and research, including mobile apps, sensors, and analytical tools.

Your work will improve researchers' ability to analyze accelerometer data for sleep monitoring and enable them to conduct large-scale studies of sleep. Ultimately, the work of this competition could improve awareness and guidance surrounding the importance of sleep. The valuable insights into how environmental factors impact sleep, mood, and behavior can inform the development of personalized interventions and support systems tailored to the unique needs of each child.



This is a Code Competition. Refer to Code Requirements for details.

##  Evaluation Metric:
Submissions are evaluated on the average precision of detected events, averaged over timestamp error tolerance thresholds, averaged over event classes.

Detections are matched to ground-truth events within error tolerances, with ambiguities resolved in order of decreasing confidence. For both event classes, we use error tolerance thresholds of 1, 3, 5, 7.5, 10, 12.5, 15, 20, 25, 30 in minutes, or 12,  36,  60,  90, 120, 150, 180, 240, 300, 360 in steps.

You may find Python code for this metric here: Event Detection AP.

#### Detailed Description
Evaluation proceeds in three steps:

    Assignment - Predicted events are matched with ground-truth events.
    Scoring - Each group of predictions is scored against its corresponding group of ground-truth events via Average Precision. 
    Reduction - The multiple AP scores are averaged to produce a single overall score.

#### Assignment

For each set of predictions and ground-truths within the same event x tolerance x series_id group, we match each ground-truth to the highest-confidence unmatched prediction occurring within the allowed tolerance.

Some ground-truths may not be matched to a prediction and some predictions may not be matched to a ground-truth. They will still be accounted for in the scoring, however.

#### Scoring

Collecting the events within each series_id, we compute an Average Precision score for each event x tolerance group. The average precision score is the area under the precision-recall curve generated by decreasing confidence score thresholds over the predictions. In this calculation, matched predictions over the threshold are scored as TP and unmatched predictions as FP. Unmatched ground-truths are scored as FN.

#### Reduction

The final score is the average of the above AP scores, first averaged over tolerance, then over event.

Submission File

For each series indicated by series_id, predict each event occurring in that series by giving the event type and the step where the event occurred as well as a confidence score for that event. The evaluation metric additionally requires a row_id with an enumeration of events.
The file should contain a header and have the following format:

    row_id,series_id,step,event,score
    0,038441c925bb,100,onset,0.0
    1,038441c925bb,105,wakeup,0.0
    2,03d92c9f6f8a,80,onset,0.5
    3,03d92c9f6f8a,110,wakeup,0.5
    4,0402a003dae9,90,onset,1.0
    5,0402a003dae9,120,wakeup,1.0
    ...

Note that while the ground-truth annotations were made following certain conventions (as described on the Data page), there are no such restrictions on your submission file.

##  Dataset Description:
The dataset comprises about 500 multi-day recordings of wrist-worn accelerometer data annotated with two event types: onset, the beginning of sleep, and wakeup, the end of sleep. Your task is to detect the occurrence of these two events in the accelerometer series.

While sleep logbooks remain the gold-standard, when working with accelerometer data we refer to sleep as the longest single period of inactivity while the watch is being worn. For this data, we have guided raters with several concrete instructions:

    A single sleep period must be at least 30 minutes in length
    A single sleep period can be interrupted by bouts of activity that do not exceed 30 consecutive minutes
    No sleep windows can be detected unless the watch is deemed to be worn for the duration (elaborated upon, below)
    The longest sleep window during the night is the only one which is recorded
    If no valid sleep window is identifiable, neither an onset nor a wakeup event is recorded for that night.
    Sleep events do not need to straddle the day-line, and therefore there is no hard rule defining how many may occur within a given period. However, no more than one window should be assigned per night. For example, it is valid for an individual to have a sleep window from 01h00–06h00 and 19h00–23h30 in the same calendar day, though assigned to consecutive nights
    There are roughly as many nights recorded for a series as there are 24-hour periods in that series.

Though each series is a continuous recording, there may be periods in the series when the accelerometer device was removed. These period are determined as those where suspiciously little variation in the accelerometer signals occur over an extended period of time, which is unrealistic for typical human participants. Events are not annotated for these periods, and you should attempt to refrain from making event predictions during these periods: an event prediction will be scored as false positive.

Each data series represents this continuous (multi-day/event) recording for a unique experimental subject.

Note that this is a Code Competition, in which the actual test set is hidden. In this public version, we give some sample data in the correct format to help you author your solutions. The full test set contains about 200 series.

#### Files and Field Descriptions

train_series.parquet - Series to be used as training data. Each series is a continuous recording of accelerometer data for a single subject spanning many days.

    series_id - Unique identifier for each accelerometer series.
    step - An integer timestep for each observation within a series.
    timestamp - A corresponding datetime with ISO 8601 format %Y-%m-%dT%H:%M:%S%z.
    anglez - As calculated and described by the GGIR package, z-angle is a metric derived from individual accelerometer components that is commonly used in sleep detection, and refers to the angle of the arm relative to the vertical axis of the body
    enmo - As calculated and described by the GGIR package, ENMO is the Euclidean Norm Minus One of all accelerometer signals, with negative values rounded to zero. While no standard measure of acceleration exists in this space, this is one of the several commonly computed features

test_series.parquet - Series to be used as the test data, containing the same fields as above. You will predict event occurrences for series in this file. 

train_events.csv - Sleep logs for series in the training set recording onset and wake events.

    series_id - Unique identifier for each series of accelerometer data in train_series.parquet.
    night - An enumeration of potential onset / wakeup event pairs. At most one pair of events can occur for each night.
    event - The type of event, whether onset or wakeup.
    step and timestamp - The recorded time of occurence of the event in the accelerometer series.

sample_submission.csv - A sample submission file in the correct format. See the Evaluation page for more details.

train_events.csv - column name: series_id, night, event, step, timestamp


## Dataset folder Location: 
../../kaggle-data/child-mind-institute-detect-sleep-states. In this folder, there are the following files you can use: sample_submission.csv, train_events.csv, test_series.parquet, train_series.parquet

## Solution Description:
First of all, I would like to express gratitude to all participants and the competition host. It was a challenging competition, but I am pleased with the positive outcome and feel relieved.
Here is a brief summary of our solution.
You can check our code here.
Single model
Here's a log on how to improve the CV score after the summary. The final scores were: CV: 0.8206, public LB: 0.768, private LB: 0.829 (equivalent to 9th place).

### Summary

#### Model structure
The model structure is primarily based on this amazing notebook, with a structure comprising:

CNN (down sample) → Residual GRU → CNN (up sample)

Overall Structure:

GRUNET is a hybrid model with several key components:
	1.	Convolutional Encoder Layers (from EncoderLayer),
	2.	Residual BiGRU Layers (from ResidualBiGRU),
	3.	Transposed Convolutional Decoder Layers (from dconv),
	4.	Final 1D Convolution Layer (output_layer).

The architecture can be broken down into the following stages:

1. Convolutional Encoder Block:
	•	The model starts with a sequence of convolutional layers defined by the arch parameter.
	•	Each layer in the arch list is a tuple (in_channels, out_channels, stride, kernel_size) that specifies the input/output channels, stride, and kernel size for the convolution operation.
	•	These convolutional layers are implemented in the conv block, which consists of multiple layers stacked together:
	•	EncoderLayer processes the input data and reduces its dimensions (based on the stride and kernel size).
	•	The final output of this block will be the feature maps that the network will use for further processing.

2. Residual BiGRU Layers:
	•	After the convolutional encoding, the data is passed through a series of residual Bi-directional GRU (BiGRU) layers.
	•	The ResidualBiGRU module contains:
	•	A bidirectional GRU that processes the sequential data (in both forward and backward directions).
	•	A series of fully connected layers and layer normalization for processing the hidden states of the GRU.
	•	A residual connection is added to the output of each BiGRU to help preserve information from earlier stages in the network.
	•	These layers are stacked sequentially in the res_bigrus module list, and the number of layers is defined by the n_layers parameter.

3. Transposed Convolutional Decoder Block (Deconvolution):
	•	After the BiGRU layers, the model uses a transposed convolutional layer (also known as a deconvolution) to upsample the data. This is implemented in the dconv block.
	•	The transposed convolution operation increases the spatial resolution of the feature maps and attempts to reconstruct or smooth the features to match the original input shape (or a desired output shape).
	•	This block is a sequence of multiple transposed convolution operations, each followed by standard convolution operations for refinement.

4. Final Convolutional Layer:
	•	After the decoder block, the final output is processed by a 1D Convolution layer, which is defined as output_layer. This layer helps to adjust the output channel dimensions to a final form suitable for prediction or classification.

Detailed Parameters and Key Variables:
	•	Input size: The model takes input with in_channels (e.g., 2 channels) and processes them through the encoder and recurrent layers.
	•	Convolutional Layer Architecture (arch):
	•	The arch parameter defines the sequence of convolutional layers. For example:
	•	(2, 8, 2, 17) means that the first layer takes 2 channels as input, outputs 8 channels, uses a stride of 2, and a kernel size of 17.
	•	The other layers follow similar definitions, with different input-output channel sizes, strides, and kernel sizes.
	•	Hidden Size: The hidden_size is specified as the output channels of the last convolutional layer (arch[-1][1]). This hidden size is passed to each of the BiGRU layers.
	•	Number of Layers (n_layers): The n_layers parameter defines how many BiGRU layers are stacked in the model.
	•	Bidirectional GRU (bidir): The model uses a bidirectional GRU for processing sequences in both forward and backward directions.
	•	Dilation: The model uses dilated convolutions for the encoder and decoder layers, allowing the network to capture larger receptive fields without increasing the number of parameters.

Key Operations in the Forward Pass:
	1.	Input Preprocessing:
	•	The input x is passed through the convolutional encoder layers (self.conv).
	2.	Residual BiGRU Layers:
	•	After convolutional encoding, the output is passed sequentially through the stacked ResidualBiGRU layers, where each layer refines the sequence representations while maintaining the residual connections.
	3.	Upsampling with Transposed Convolutions:
	•	The feature map is upsampled through the transposed convolution layers (self.dconv).
	4.	Final Output Layer:
	•	The result is passed through a final 1D convolution (self.output_layer) to produce the final output.

Summary of Flow:
	•	Input → Convolutional Encoder Layers → Residual BiGRU Layers → Transposed Convolutional Decoder → Final 1D Convolution Output.



SEScale

For input scaling, SEModule was utilized. (https://arxiv.org/abs/1709.01507)

class SEScale(nn.Module):
   def __init__(self, ch: int, r: int) -> None:
       super().__init__()
       self.fc1 = nn.Linear(ch, r)
       self.fc2 = nn.Linear(r, ch)

   def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
       h = self.fc1(x)
       h = F.relu(h)
       h = self.fc2(h).sigmoid()
       return h * x


Minute connection

As noted in several discussions and notebooks, there was a bias in the minute when ground truth events occurred. To account for this, features related to minutes were concatenated separately in the final layer.

   def forward(self, num_x: torch.FloatTensor, cat_x: torch.LongTensor) -> torch.FloatTensor:
       cat_embeddings = [embedding(cat_x[:, :, i]) for i, embedding in enumerate(self.category_embeddings)]
       num_x = self.numerical_linear(num_x)

       x = torch.cat([num_x] + cat_embeddings, dim=2)
       x = self.input_linear(x)
       x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)

       for gru in self.gru_layers:
           x, _ = gru(x)

       x = self.dconv(x.transpose(-1, -2)).transpose(-1, -2)
       minute_embedding = self.minute_embedding(cat_x[:, :, 0])
       x = self.output_linear(torch.cat([x, minute_embedding], dim=2))
       return x

Data Preparation

Each series of data is divided into daily chunks, offset by 0.35 days.

       train_df = train_df.with_columns(pl.arange(0, pl.count()).alias("row_id"))
       series_row_ids = dict(train_df.group_by("series_id").agg("row_id").rows())

       series_chunk_ids = []  # list[str]
       series_chunk_row_ids = []  # list[list[int]]
       for series_id, row_ids in tqdm(series_row_ids.items(), desc="split into chunks"):
           for start_idx in range(0, len(row_ids), int(config.stride_size / config.epoch_sample_rate)):
               if start_idx + config.chunk_size <= len(row_ids):
                   chunk_row_ids = row_ids[start_idx : start_idx + config.chunk_size]
                   series_chunk_ids.append(series_id)
                   series_chunk_row_ids.append(np.array(chunk_row_ids))
               else:
                   chunk_row_ids = row_ids[-config.chunk_size :]
                   series_chunk_ids.append(series_id)
                   series_chunk_row_ids.append(np.array(chunk_row_ids))
                   break

During training, half of each chunk is used in every epoch.

               sampled_train_idx = train_idx[epoch % config.epoch_sample_rate :: config.epoch_sample_rate]

For evaluation, overlapping sections are averaged, and the ends of each chunk are trimmed by 30 minutes.

Target

A decaying target is created based on the distance from the ground truth event, with diminishing values as the distance increases.

tolerance_steps = [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
target_columns = ["event_onset", "event_wakeup"]

train_df = (
   train_df.join(train_events_df.select(["series_id", "step", "event"]), on=["series_id", "step"], how="left")
   .to_dummies(columns=["event"])
   .with_columns(
       pl.max_horizontal(
           pl.col(target_columns)
           .rolling_max(window_size * 2 - 1, min_periods=1, center=True)
           .over("series_id")
           * (1 - i / len(tolerance_steps))
           for i, window_size in enumerate(tolerance_steps)
       )
   )
)

The target is updated each epoch to decay further.

               # update target
               targets = np.where(targets == 1.0, 1.0, (targets - (1.0 / config.n_epochs)).clip(min=0.0))

By attenuating the target, the range of predicted values narrows, allowing for the detection of finer peaks.

Periodicity Filter

As discussed in here, there is a daily periodicity in the data when the measuring device is removed. This is leveraged to predict these periods rule-based and used as a filter for inputs and predictions.

Input Features

categorical features

    hour
    minute
    weekday
    periodicity flag

numerical features

    anglez / 45
    enmo.log1p().clip_max(1.0) / 0.1
    anglez, enmo 12 steps rolling_mean, rolling_std, rolling_max
    anglez_diff_abs 5 min rolling median

Change logs

    baseline model (cv: 0.7510) - public: 0.728
    Add a process to decay the target every epoch (cv: 0.7699, +19pt)
    Add a periodicity filter to the output (cv: 0.7807, +11pt)
    Add a periodicity flag to the input as well (cv: 0.7870, +6pt) - public: 0.739
    batch_size: 16 → 4, hidden_size: 128 → 64, num_layers: 2 → 8 (cv: 0.7985, +11pt) - public: 0.755
    Normalize the score in the submission file by the daily score sum (cv: 0.8044, +6pt)
    Remove month and day from the input (cv: 0.8117, +7pt)
    Trim the edges of the chunk by 30 minutes on both sides (cv: 0.8142, +4pt) - public: 0.765
    Modify to concatenate the minute features to the final layer (cv: 0.8206, +6pt) - public: 0.768


Post Processing

This post-processing creates a submission DataFrame to optimize the evaluation metrics. With this post-processing method, we significantly improved our scores (public: 0.768 → 0.790, private: 0.829 → 0.852 !!!).
This was a complex procedure, which I will explain step by step.

1.Data Characteristics

First, let's discuss the characteristics of the data. As noted in several discussions and notebooks, the second of the target events was always set to zero.
The competition's evaluation metric doesn't differentiate predictions within a 30-second range from the ground truth event. So, whether the submission timestamp's seconds are 5, 10, 15, 20, … 25, the same score is returned.

2.Creation of the 2nd Level Model

The 1st level model's predictions were trained to recognize events within a certain range from the ground truth as positive. However, the 2nd level model transforms these into probabilities of a ground truth event existing for each minute.
The output of the 1st level model was at seconds 0, 5, 10, …, but the 2nd level model aggregates these to always be at second 0. Specifically, it inputs aggregated features around hh:mm:00 and learns to predict 1 only at the exact time of an event, otherwise 0. Details of the 2nd level model will be described later.

3.Score Calculation for Each Point
As explained earlier, submitting any second within the same minute yields the same score. Therefore, we estimate the score at the 15 and 45 second points of each minute, and submit the one with the highest value, effectively submitting the highest score for all points. The method of score estimation is as follows:
For instance, let's estimate the score at 10:00:15.

First, we create a window of 12 steps from the point of interest and sum the predictions of the 2nd level model within this window to calculate the tolerance_12_score.

Similarly, we calculate tolerance_36_score, tolerance_60_score, …, for the respective tolerances used in the evaluation, and the sum of these scores is considered the score for the point of interest.

We perform this calculation for all points, and for each series, we adopt the point with the highest score and add it to the submission DataFrame.

4.Score Recalculation

Next, we recalculate the score to determine the next point to be adopted. For example, suppose the point 09:59:15 was chosen.
First, consider updating the tolerance_12_score. Events within tolerance 12 of the adopted point cannot match overlappingly with the next point to be submitted.

Therefore, when calculating the tolerance_12_score for the next point to be adopted, it's necessary to discount the prediction values within 12 steps of the currently adopted point.

Likewise, for tolerance_36_score, tolerance_60_score, …, we recalculate by discounting the prediction values within 36, 60, …, steps of the adopted point.

With the updated scores calculated, we again adopt the highest scoring point for each series and add it to the submission dataframe.

5.Creating Submissions
We repeat the above step 4 to extract a sufficient number of submission points, then compile these into a DataFrame to create the submission file.

Additional Techniques

Several other techniques were employed to make the post-processing work effectively:

Normalize the predictions of the 2nd level model daily.
When recalculating the score, calculate the difference from the previous score to reduce the computation.
Speed up the above calculations using JIT compilation.

Details of the 2nd Level Model

The 2nd level model starts by averaging the 1st level model's predictions on a per-minute basis and then detecting peaks in these averages using find_peaks with a height of 0.001 and a distance of 8.
Based on the detected peaks, chunks are created from the original time series, capturing 8 minutes before and after each peak. (Recall: 0.9845)
This step_size was crucial because the ratio of positive to negative examples changes depending on how many steps are included, affecting the accuracy of subsequent stages. Therefore, we tuned the number of steps for optimal performance.
If chunks are connected, they are treated as a single chunk.


For each chunk, we aggregated features from the 1st model's predictions and other features like anglez and enmo. These aggregated features were then used to train models such as LightGBM and CatBoost.

Additionally, we treated each chunk as a sequence for training CNN-RNN, CNN, and Transformer models. As a result, we developed a model that could account for minute biases not fully addressed by the 1st level model.

The predictions of the 2nd level model were sufficiently calibrated, so there was no need for further transformation..


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: