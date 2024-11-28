# 🔍Private code for Fairness Analysis for Face Verification 📊

***

## ✨ Overview
This repository provides tools for conducting a comprehensive fairness analysis of face verification models. It is part of the study presented in the paper:
**[Fairer Analysis and Demographically Balanced Face Generation for Fairer Face Verification](LINK HERE)**.

### Features
* Computes **basic fairness metrics** (from [Fairlean](https://fairlearn.org/))
* Performance a **variance analysis of the model's latent space**.
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
***
## 🗂️ Supported Datasets
This script supports the following datasets for evaluation:
- RFW: [Racial Faces in the Wild](http://whdeng.cn/RFW/testing.html)
- BFW: [Balanced Faces in the Wild](https://ieee-dataport.org/documents/balanced-faces-wild)
- FAVCI2D: [Face Verification with Challenging Imposters and Diversified Demographics](https://github.com/AIMultimediaLab/FaVCI2D-Face-Verification-with-Challenging-Imposters-and-Diversified-Demographics)
***
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

***

## ⚙️ Example Usage
Run the analysis using a single command: 
```bash
python compute_metrics.py --dataset=rfw --results_path=model_results/BUPT_RFW.csv
```
Use the `--alpha` flag ro modify the 95% confidence interval (default is 0.05 for 95% confidence intervals).
### Input Format
The results file must be a CSV with the following columns:
- `img_1`: filename of the first image in the pair
- `img_2`: filename of the second image in the pair
- `dist`: L2 distance between the embeddings of the two images (_automatically converted to angles_).

***

## 🛠️ Setup and Installation
To install dependencies, run:
```bash
pip install -r requirements.txt
```
Ensure the `data/` directory is populated with the necessary demographic attributes before running the script.


***

## 🙌 Acknowledgments
Special thanks to the developers of [Fairlearn](https://fairlearn.org/), [FairFace](https://github.com/joojs/fairface), and [Statsmodels](https://www.statsmodels.org/stable/index.html) for their invaluable tools and resources.