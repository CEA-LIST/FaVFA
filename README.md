# Private code for Fairness Analaysis for Face Verification

## TO DO:
* [ ] Create processed image metadata CSV files
* [X] Create processed pair metadata CSV files
* [X] Create code to get metrics from Table 2
* [ ] Create code to get ANOVA results (R2 and eta-squared)
* [ ] Create code to get fairness marginal effects in TPR and FPR with confidence intervals

## Usage
`
python compute_metrics.py --dataset=rfw --results_path=model_results/BUPT_RFW.csv
`