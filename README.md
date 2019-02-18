# Multi-digit detection  via attention and classification

### Preparing training and testing data
First we need to generate the valid pixel locations.

1. Go to the cloned multi_digit_classification_attention folder and run the following command:
``` bash
mkdir dataset
```

2. Download the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset and extract the train and test SVHN data into the *dataset* folder inside *multi_digit_classification_attention* folder.

3. Select which type to data to curate and run the following command:
``` bash
python crop_dataset --dataset_type=<train/test>
```
