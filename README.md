# T4SA-2.0

In this repository there is all the main material related to my master degree thesis ***"Cross-modal learning for sentiment analysis of social media images"***.
Please, check the [Documentation](Documentation.pdf) for a complete view. 

This work introduced the **T4SA 2.0 dataset**, i.e. a big set of data to train visual models for **Sentiment Analysis in the Twitter domain**.

It was used to fine-tune the Vision-Transformer (ViT) model pre-trained on Imagenet-21k, which was able to achieve incredible results on external benchmarks which were manually annotated, even ***beating the current State Of The Art!***

It was built crawling ∼3.7 pictures from the social media during the continuous period 1 April-30 June and then labeling them using a **cross-modal** approach. In particular, a distant supervised technique was used to get rid of human annotators, thus minimizing the efforts required and allowing for the creation of a huge dataset.

The latter can help future research to train robust visual models. Its size would be particularly advantageous as the number of parameters of current SOTA is **exponentially growing along with their need of data** to avoid overfitting problems.


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

## How to use script for benchmark evaluation
### Test a model with a benchmark, get the accuracy and save the prediction
    $ python3 python\ script/test_benchmark.py -m <model_name> -b <benchmark_name>
### Execute a five fold cross validation on a benchmark, get the mean accuracy, the standard deviation and save the predictions (by default use the boosted_model)
    $ python3 python\ script/5_fold_cross.py -b <benchmark_name>
### Fine tune FI on the five split, get the mean accuracy, the standard deviation and save the predictions (by default use the boosted_model)
    $ python3 python\ script/fine_tune_FI.py
    
## Directory structure
```bash
.
├── colab notebook
├── dataset
│   ├── benchmark
│   │   ├── EmotionROI
│   │   │   └── images
│   │   │       ├── anger
│   │   │       ├── disgust
│   │   │       ├── fear
│   │   │       ├── joy
│   │   │       ├── sadness
│   │   │       └── surprise
│   │   ├── FI
│   │   │   ├── images
│   │   │   │   ├── amusement
│   │   │   │   ├── anger
│   │   │   │   ├── awe
│   │   │   │   ├── contentment
│   │   │   │   ├── disgust
│   │   │   │   ├── excitement
│   │   │   │   ├── fear
│   │   │   │   └── sadness
│   │   │   ├── split_1
│   │   │   ├── split_2
│   │   │   ├── split_3
│   │   │   ├── split_4
│   │   │   └── split_5
│   │   ├── Twitter Testing Dataset I
│   │   │   └── images
│   │   └── Twitter Testing Dataset II
│   │       └── images
│   ├── t4sa 1.0
│   │   ├── dataset with new labels
│   │   │   ├── b-t4sa1.0 updated
│   │   │   └── b-t4sa1.0 updated and filtered
│   │   └── original dataset
│   │       └── b-t4sa 1.0
│   └── t4sa 2.0
│       ├── bal_T4SA2.0
│       ├── bal_flat_T4SA2.0
│       ├── img
│       ├── merged_T4SA
│       └── unb_T4SA2.0
├── models
├── predictions
└── python script
``` 


