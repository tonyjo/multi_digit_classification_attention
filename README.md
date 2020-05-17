# Multi-character Prediction Using Attention
<p align="center">
  <img src="./images/title.png" width="700">
</p>

## Setup
``` bash
pip install -r requirements.txt
```
Codebase developed on `Python-2.7.15`

## Multi-Digit classification - (SVHN dataset)

### Preparing data
We need to prepare the raw SVHN dataset

1. Go to the cloned `multi_digit_classification_attention` folder and run the following command:
``` bash
cd SVHN
mkdir dataset
```

2. Download the [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset and extract the train and test SVHN data into the `dataset` folder inside `multi_digit_classification_attention` folder.

3. Select which type to data to curate and run the following command:
``` bash
python gen_crop_dataset.py --dataset_type=<train/test>
```

4. Select which type to data to generate attention mask and run the following command:
``` bash
python gen_attn_truth.py --dataset_type=<train/test>
```

### Training

1. To train detection model run the following command:
``` bash
python train.py
```

2. To train classification model run the following command:
``` bash
python train_classify_net.py
```

### Inference
To test and visualize results run the following command:
``` bash
jupyter notebook
```
and open and run:
``` bash
> evaluate_and_viz.ipynb
```

### End-2-End Learning for detection and classification

1. To train full model for both detection and classification run the following command:
``` bash
python train_end2end.py
```

2. To test and visualize results run the following command:
``` bash
jupyter notebook
```
and open and run:
``` bash
> evaluate_and_viz_end2end.ipynb
```


## CAPTCHA classification - (CAPTCHA dataset)

### Preparing data
We need to generate raw CAPTCHA dataset

1. Go to the cloned `multi_digit_classification_attention` folder and run the following command:
```bash
cd other/
mkdir dataset
```

2. Generate dataset by running the following command:
``` bash
python gen_captcha_dataset.py
```

3. Move the generated dataset into `CAPTCHA` folder.


### Training

1. To train detection model run the following command:
``` bash
python train.py
```

2. To train classification model run the following command:
``` bash
python train_classify_net.py
```

### Inference

To test and visualize results run the following command:
``` bash
jupyter notebook
```
and open and run:
``` bash
> evaluate_and_viz.ipynb
```

### End-2-End Learning for detection and classification

1. To train full model for both detection and classification run the following command:
``` bash
python train_end2end.py
```

2. To test and visualize results run the following command:
``` bash
jupyter notebook
```
and open and run:
``` bash
> evaluate_and_viz_end2end.ipynb
```


## Contribution guidelines

Any improvements would be appreciated, send a merge request if you would like to contribute.
