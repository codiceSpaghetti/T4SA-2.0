# T4SA-2.0

In this repository there is all the main material related to my master degree thesis ***"Cross-modal learning for sentiment analysis of social media images"***.

This work introduced the **T4SA 2.0 dataset**, i.e. a big set of data to train visual models for **Sentiment Analysis in the Twitter domain**.

It was used to fine-tune the Vision-Transformer (ViT) model pre-trained on Imagenet-21k, which was able to achieve incredible results on external benchmarks which were manually annotated, even ***beating the current State Of The Art!***

It was built crawling ∼3.7 pictures from the social media during the continuous period 1 April-30 June and then labeling them using a **cross-modal** approach. In particular, a distant supervised technique was used to get rid of human annotators, thus minimizing the efforts required and allowing for the creation of a huge dataset.

The latter can help future research to train robust visual models. Its size would be particularly advantageous as the number of parameters of current SOTA is **exponentially growing along with their need of data** to avoid overfitting problems.


