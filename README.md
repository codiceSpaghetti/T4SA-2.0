# T4SA-2.0

In this repository there is all the main material related to my master degree thesis ***"Cross-modal learning for sentiment analysis of social media images"***.
Please, check the [Documentation](Documentation.pdf) for a complete view. 

This work introduced the **T4SA 2.0 dataset**, i.e. a big set of data to train visual models for **Sentiment Analysis in the Twitter domain**.

It was used to fine-tune the Vision-Transformer (ViT) model pre-trained on Imagenet-21k, which was able to achieve incredible results on external benchmarks which were manually annotated, even ***beating the current State Of The Art!***

It was built crawling ∼3.7 pictures from the social media during the continuous period 1 April-30 June and then labeling them using a **cross-modal** approach. In particular, a distant supervised technique was used to get rid of human annotators, thus minimizing the efforts required and allowing for the creation of a huge dataset.

The latter can help future research to train robust visual models. Its size would be particularly advantageous as the number of parameters of current SOTA is **exponentially growing along with their need of data** to avoid overfitting problems.

## Directory structure
[01;34m.[00m
├── [01;34mcolab notebook[00m
├── [01;34mdataset[00m
│   ├── [01;34mbenchmark[00m
│   │   ├── [01;34mEmotionROI[00m
│   │   │   └── [01;34mimages[00m
│   │   │       ├── [01;34manger[00m
│   │   │       ├── [01;34mdisgust[00m
│   │   │       ├── [01;34mfear[00m
│   │   │       ├── [01;34mjoy[00m
│   │   │       ├── [01;34msadness[00m
│   │   │       └── [01;34msurprise[00m
│   │   ├── [01;34mFI[00m
│   │   │   ├── [01;34mimages[00m
│   │   │   │   ├── [01;34mamusement[00m
│   │   │   │   ├── [01;34manger[00m
│   │   │   │   ├── [01;34mawe[00m
│   │   │   │   ├── [01;34mcontentment[00m
│   │   │   │   ├── [01;34mdisgust[00m
│   │   │   │   ├── [01;34mexcitement[00m
│   │   │   │   ├── [01;34mfear[00m
│   │   │   │   └── [01;34msadness[00m
│   │   │   ├── [01;34msplit_1[00m
│   │   │   ├── [01;34msplit_2[00m
│   │   │   ├── [01;34msplit_3[00m
│   │   │   ├── [01;34msplit_4[00m
│   │   │   └── [01;34msplit_5[00m
│   │   ├── [01;34mTwitter Testing Dataset I[00m
│   │   │   └── [01;34mimages[00m
│   │   └── [01;34mTwitter Testing Dataset II[00m
│   │       └── [01;34mimages[00m
│   ├── [01;34mt4sa 1.0[00m
│   │   ├── [01;34mdataset with new labels[00m
│   │   │   ├── [01;34mb-t4sa1.0 updated[00m
│   │   │   └── [01;34mb-t4sa1.0 updated and filtered[00m
│   │   └── [01;34moriginal dataset[00m
│   │       └── [01;34mb-t4sa 1.0[00m
│   └── [01;34mt4sa 2.0[00m
│       ├── [01;34mbal_T4SA2.0[00m
│       ├── [01;34mbal_flat_T4SA2.0[00m
│       ├── [01;34mimg[00m
│       ├── [01;34mmerged_T4SA[00m
│       └── [01;34munb_T4SA2.0[00m
├── [01;34mmodels[00m
├── [01;34mpredictions[00m
└── [01;34mpython script[00m

45 directories

## Build Instructions
### Install git
    $ apt-get update
    $ apt-get install git
### Install T4SA codebase
    $ git config --global http.postBuffer 1048576000
    $ git clone --recursive https://github.com/codiceSpaghetti/T4SA-2.0.git
### Install dependencies 
    $ chmod +x install_dependencies.sh
    $ ./install_dependencies.sh
##  Download models
    $ chmod +x download_models.sh
    $ ./download_models.sh
### Test a model with a benchmark and get the accuracy
    $ python3 python\ script/test_benchmark.py -m <model_name> -b <benchmark_name>
### Execute a five fold cross validation on a benchmark and get the mean accuracy and standard deviation (by default use the boosted_model)
    $ python3 python\ script/5_fold_cross.py -b <benchmark_name>
### Fine tune FI on the five split and get the mean accuracy and standard deviation (by default use the boosted_model)
    $ python3 python\ script/fine_tune_FI.py


