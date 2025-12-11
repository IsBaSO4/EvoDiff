  <h2> Generative Active Learning for Label-Efficient Pan-Organ Ultrasound Diagnosis </h2>

## 📋 Introduction

In this work, we first proposed a universal active learning (AL) framework that combines **Evo**lutionary optimization with information gain guidance during **Diff**usion sampling, named EvoDiff. By reformulating sample selection as a diffusion-based feature space optimization problem, EvoDiff bridges diffusion sampling trajectories and evolutionary search for efficient data annotation in medical ultrasound imaging.


## ✨ Key Features

- **🎯 Information Gain Estimation**: Calculates entropy differentials along DDIM sampling paths to guide generation toward informative regions without backpropagation
- **🧬 Evolutionary Optimization**: Uses evolutionary perturbations to escape local optima and explore diverse feature spaces
- **📚 Natural Curriculum Learning**: Progressively generates samples from certain to uncertain cases, yielding smoother learning dynamics
- **🔧 Architecture Agnostic**: Compatible with both CNNs and Transformer-based foundation models
- **⚡ High Efficiency**: 24–84% reduction in manual annotation time compared to conventional workflows

## 🏥 Supported Applications

EvoDiff has been validated across multiple ultrasound diagnostic tasks:

- ✅ Appendix diagnosis
- ✅ Breast tumor recognition
- ✅ Cardiovascular event prediction  
- ✅ Fatty liver diagnosis
- ✅ Ovarian tumor assessment
- ✅ Thyroid nodule diagnosis

## 📊 Performance Highlights

Our comprehensive evaluation demonstrates:

- **Dataset Coverage**: 6 benchmarks, 12 datasets with 16,482 images
- **Clinical Validation**: 8 internal cohorts + 4 independent external cohorts
- **Label Efficiency**: Approaches fully supervised performance with substantially fewer labels
- **State-of-the-Art**: Outperforms existing AL methods across multi-organ ultrasound tasks
- **Generalization**: Enhanced zero-shot performance on unseen external datasets
- **Clinical Impact**: Validated in radiologist reader study with multiple experience levels


## 🚀 Getting Started

#### 1. Clone the Repository

```
git clone https://github.com/IsBaSO4/EvoDiff.git
cd EvoDiff
```

#### 2. Environment Setup

First install pytorch according to [guided-diffusion](https://github.com/openai/guided-diffusion).
```
# install requirments
pip install -r requirements.txt
```

#### 3. Data preparation

Download datasets:

The public medical datasets used in this study are accessible from the following sources: [Appendix](https://zenodo.org/records/7669442), [BUSI](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset), [BUS-BRA](https://zenodo.org/records/8231412), [UDIAT](https://ieeexplore.ieee.org/abstract/document/8003418), [ALN-Ultra](https://zenodo.org/records/15003119), [BUS_UC](https://data.mendeley.com/datasets/3ksd7w7jkx/1), [QAMEBI](https://qamebi.com/breast-ultrasound-images-database), [CUBS](https://data.mendeley.com/datasets/fpv535fss7/1), [Fatty-Liver](https://zenodo.org/records/1009146), [MMOTU](https://figshare.com/articles/dataset/_zip/25058690?file=44222642), [TN3K](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), and [TN5000](https://figshare.com/s/cb6a67f17c04b29e7edd).


Training data should be saved in the following form:
```
datasets/
  ├── dataset_1/
    ├── 0/ img1.png..
    ├── 1/ img1.png..
  ├── dataset_2/
    ├── 0/ img1.png..
    ├── 1/ img1.png..
  ├──...
```

## Step 1: Pre-training

This step involves training two models and must be executed twice with different datasets.

### Training Process Overview

**First Iteration**: Train on unlabeled data  
**Second Iteration**: Train on labeled data

### 1. Diffusion Model

```Python
(CUDA_VISIBLE_DEVICES=$device ) mpiexec -n $gpu_num python image_train.py  # multi-gpu parallel
```

### 2. Gudiance Classifier Model

```Python
(CUDA_VISIBLE_DEVICES=$device ) mpiexec -n $gpu_num python classifier_train.py  # multi-gpu parallel
```

## Step 2: Information Gain-guided Sampling

#### Information Gain with Evolutionary Algorithm Generation
```Python
(CUDA_VISIBLE_DEVICES=$device )python ig_with_ea_sample.py --single_gpu True  --num_classes 13  --SAMPLE_CLASSIFIER_SCALE $1 # specific single gpu(default is 0)
```

```
--SAMPLE_CLASSIFIER_SCALE           # Information Gain Gudiance Strength. Default is 1.
```


## Step 3: Sample Seletion with Curriculum Learning

#### Sample Selection
```Python
(CUDA_VISIBLE_DEVICES=$device )python class_query_select.py # specific single gpu(default is 0)
```

## Step 4: Downstream Models Initialization and Fine-tuning.

Pre-trained weights and implementation code for the downstream models are accessible through the following repositories: [USFM](https://github.com/openmedlab/USFM) and [ResNet-50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py).



## Acknowledgements

We sincerely appreciate the code release of the following projects: [guided-diffusion](https://github.com/openai/guided-diffusion), [improved-diffusion](https://github.com/openai/improved-diffusion) and [UGDM](https://github.com/yangqy1110/MGDM/tree/main).