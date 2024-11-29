# 🔍Private code for Fairness Analysis for Face Verification 

This repository provides tools for conducting a comprehensive fairness analysis of face verification models. It is part of the study presented in the paper
**Fairer Analysis and Demographically Balanced Face Generation for Fairer Face Verification** published at WACV 2025 (see [credits](#-acknowledgments-and-credits))

If you are rather intersted by the fair dataset of synthetic faces and the code to generate one, look at [this repository](https://github.com/afm215/FaVGen-dev).

It is [easy to install](#️-setup-and-installation) with few dependencies and [easy to use](#️-example-usage) on [three face academic verification benchmarks](#️-supported-datasets)

## Table of Contents
1. [✨ Overview](#-overview)
2. [🗂️ Supported Datasets](#️-supported-datasets)
3. [📏 Computed Metrics](#computed-metrics)
4. [⚙️ Example Usage](#️-example-usage)
5. [🛠️ Setup and Installation](#️-setup-and-installation)
6. [🙌 Acknowledgments and Credits](#-acknowledgments-and-credits)

## ✨ Overview
This code implements a method to estimate to which extent a face verification method is *fair* that is whether its performance are the same for e.g *male* and *female* persons, do not depend on the *age* of the person or its *ethnicity*.

The task of face verification determines whether two face images represent the same person. Given its score on [some academic benchmarks](#️-supported-datasets), our method computes [several fainess metrics](#computed-metrics) then quantifies to which extend a particular group (e.g *female*) is better/less well recognized than another one (e.g *male*).

### Features
* Computes **basic fairness metrics** (from [Fairlean](https://fairlearn.org/))
* Performs a **variance analysis of the model's latent space**.
* Evaluates the **marginal effects of demographic attributes** (e.g., ethnicity, gender, age) on key metrics such as True Positive Rate (TPR) and False Positive Rate (FPR).

### Demographic attributes
The analysis uses precomputed demographic attributes stored in  `data/`. The following attributes are considered:
- **Ethnicity**: ethnicities in the images pair (e.g. `White x White`). Provided or inferred using the [FairFace](https://github.com/joojs/fairface) model.
- **Gender**: genders in the images pair (e.g. `Male x Male`) provided or inferred using [FairFace](https://github.com/joojs/fairface).
- **Age**: age difference in the pair (continuous value). Inferred using [FairFace](https://github.com/joojs/fairface).
- **Pose**: relative position between the position of the two faces. Computed using TODO. Encoded using either:
  - `angle` (angle between the position vectors).
  - `x_dist`,`y_dist`,`z_dist`: distance variables along each spacial dimension. 

**Note**: Negative pairs with obvious demographic differences (e.g., different ethnicities or genders) are filtered out. The analysis focuses on "**hard**" negative pairs, as detailed in the paper.

## 🗂️ Supported Datasets
This script supports the following datasets for evaluation:
- RFW: [Racial Faces in the Wild](http://whdeng.cn/RFW/testing.html)
- BFW: [Balanced Faces in the Wild](https://ieee-dataport.org/documents/balanced-faces-wild)
- FAVCI2D: [Face Verification with Challenging Imposters and Diversified Demographics](https://github.com/AIMultimediaLab/FaVCI2D-Face-Verification-with-Challenging-Imposters-and-Diversified-Demographics)

For the standard list of pairs, the corresponding attribute labels are pre-computed and savec in csv files that can be found in `data/`.

## 📏Computed metrics
The script computes the following metrics, in order:<br>
* 1️⃣ **Basic metrics**:
  * Micro-avg Accuracy
  * Macro-avg Accuracy
  * TPR (True Positive Rate)
  * FPR (False Positive Rate)

* 2️⃣ **Fairness metrics** (using [Fairlearn](https://fairlearn.org/)):
  * Demographic Parity Difference
  * Demographic Parity Ratio
  * Equalized Odds Difference
  * Equalized Odds Ratio

* 3️⃣ **Latent space analysis (ANOVA)** (using [statsmodels](https://www.statsmodels.org/stable/index.html)):
  * Computed separately for positive and negative pairs
  * **% Explained Variance** (partial $\eta^2$).
  * **Significance Tests** (p-values)
* 4️⃣ **Marginal effects** (using [statsmodels](https://www.statsmodels.org/stable/index.html)):
  Using a logistic regression model, this computes:
  * Marginal effect of demographic attributes on **TPR** and **FPR**.
  * Outputs include:
    * Marginal effect value.
    * 95% Confidence Interval (modifiable via `--alpha`).
    * Significance p-value.

## ⚙️ Example Usage
Run the analysis using a single command: 
```bash
python compute_metrics.py --dataset=rfw --model_dist=model_results/BUPT_RFW.csv
```
Your face verification method must be tested on one of the available benchmarks, specified by `--dataset`. [Available benchmarks](#️-supported-datasets) are `bfw`, `favcid` and `rfw`. 

Your face verification method should be run on the standard testing image pairs (two first columns in `data/xxx.csv`). The resulting distances for each pair has to be saved in a CSV file with the following columns:
- `img_1`: filename of the first image in the pair
- `img_2`: filename of the second image in the pair
- `dist`: L2 distance between the embeddings of the two images (_automatically converted to angles_).

Let specity the path to this file with the flag `--model_dist`. We provide such files in `model_results/`, corresponding to the approach we proposed in our paper.

Use the `--alpha` flag to modify the 95% confidence interval (default is 0.05 for 95% confidence intervals).

## 🛠️ Setup and Installation
To install dependencies, run:
```bash
pip install -r requirements.txt
```
Ensure the `data/` directory is populated with the necessary demographic attributes before running the script.



## 🙌 Acknowledgments and Credits
Special thanks to the developers of [Fairlearn](https://fairlearn.org/), [FairFace](https://github.com/joojs/fairface), and [Statsmodels](https://www.statsmodels.org/stable/index.html) for their invaluable tools and resources.

If you find this work useful and use it on your own research, please cite our paper
```
@inproceedings{afm2025fairer_analysis,
  author = {Fournier-Montgieux, Alexandre and Soumm, Michael and Popescu, Adrian and Luvison, Bertrand and Le Borgne, Herv{\'e}},
  title = {Fairer Analysis and Demographically Balanced Face Generation for Fairer Face Verification},
  booktitle = {Winter Conference on Applications of Computer Vision (WACV)"},
  address = "Tucson, Arizona, USA",
  year = {2025},
}
```


