You are given a detailed description of a machine learning task, including the task description, evaluation metrics, dataset details, and a detailed description of a solution.
You should strictly follow the provided solution description to implement it and generate the corresponding code to complete this machine learning task.
You are required to complete this competition using Python. We will provide GPU computing resources, and you can use GPU acceleration for training.
You can use any libraries that might be helpful.
Finally, you need to generate a submission file as specified. The submission file should be named Kannada_MNIST_submission.csv. You should save it to the folder: ../submission_folder.

## Task Description
Bored of MNIST?

The goal of this competition is to provide a simple extension to the classic MNIST competition we're all familiar with. Instead of using Arabic numerals, it uses a recently-released dataset of Kannada digits.

Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script. Wikipedia

This competition uses the same format as the MNIST competition in terms of how the data is structured, but it's different in that it is a synchronous re-run Kernels competition. You write your code in a Kaggle Notebook, and when you submit the results, your code is scored on both the public test set, as well as a private (unseen) test set.

Technical Information

All details of the dataset curation has been captured in the paper titled: Prabhu, Vinay Uday. "Kannada-MNIST: A new handwritten digits dataset for the Kannada language." arXiv preprint arXiv:1908.01242 (2019)

The github repo of the author can be found here.

On the originally-posted dataset, the author suggests some interesting questions you may be interested in exploring. Please note, although this dataset has been released in full, the purpose of this competition is for practice, not to find the labels to submit a perfect score.

In addition to the main dataset, the author also disseminated an additional real world handwritten dataset (with 10k images), termed as the 'Dig-MNIST dataset' that can serve as an out-of-domain test dataset. It was created with the help of volunteers that were non-native users of the language, authored on a smaller sheet and scanned with different scanner settings compared to the main dataset. This 'dig-MNIST' dataset serves as a more difficult test-set (An accuracy of 76.1% was reported in the paper cited above) and achieving ~98+% accuracy on this test dataset would be rather commendable.


##  Evaluation Metric:
This competition is evaluated on the categorization accuracy of your predictions (the percentage of images you get correct).

Submission File Format

The file should contain a header and have the following format:

    id,label
    1,5
    2,5
    3,5
    ···

##  Dataset Description:
The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine, in the Kannada script.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, train.csv, has 785 columns. The first column, called label, is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixel{x}, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixel{x} is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:

    000 001 002 003 ... 026 027
    028 029 030 031 ... 054 055
    056 057 058 059 ... 082 083
    |   |   |   |  ...  |   |
    728 729 730 731 ... 754 755
    756 757 758 759 ... 782 783 

The test data set, test.csv, is the same as the training set, except that it does not contain the label column.

The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that you have correctly classified all but 3% of the images.

#### Files

    train.csv - the training set
    test.csv - the test set
    sample_submission.csv - a sample submission file in the correct format
    Dig-MNIST.csv - an additional labeled set of characters that can be used to validate or test model results before submitting to the leaderboard

## Dataset folder Location: 
../../kaggle-data/Kannada-MNIST. In this folder, there are the following files you can use: train.csv, test.csv, sample_submission.csv, Dig-MNIST.csv

## Solution Description:
Congratulations to all participants. The main goal for me in this competition was checking the effective way of using new optimizes like RAdam and Over9000. Despite those optimizers are promised to work much better than Adam in corresponding studies, my initial tests of these optimizers were quite disappointing: RAdam gave about the same result as Adam while Over9000 worked much worse. In the last competition on Cloud Segmentation I participated in gave quite interesting observation: if RAdam is used there must be no warm-up in cosine annealing. Over9000 worked even better with such scheduler than RAdam as expected from here. With fast.ai it can be done with the following additional arguments learn.fit_one_cycle(36, max_lr=slice(0.2e-2,1e-2), pct_start=0.0, div_factor=100). So, this competition has provided quite small dataset and fast training to experiment with optimizes and schedulers.

In this competition I used 3 models:

    1) Resnet20 from CIFAR-pretrained-models (and the corresponding dataset) on 32x32x1 images (I have replaced first conv summing corresponding pretrained weights). You can load resnet20 by this code:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    2,3) Pretrained DenseNet121 and DenseNet169 on 64x64x1 images (because of the first layer with stride 2 conv followed by 3x3 pooling it is really helpful to upscale small images). You can load these models by using torchvision.

In all models ReLU is replaced by Mish. The head is similar to one used by fast.ai by default: Concat pooling + Mish + BN + Dropout(0.5) + Linear + Mish + BN + Dropout(0.5) + Linear.

Other things used:

    Over9000 one cycle without warm-up scheduler
    Categorical Cross Entropy
    gradient clipping
    best model selection
    discriminative lr with split of the model into backbone and head
    standard fast.ai augmentation without flip get_transforms(do_flip=False,max_zoom=1.2)
    5 fold CV and ignore public LB


Just directly write the code and include necessary comments. Don't output other thing. Now write you code here. 
Your code: