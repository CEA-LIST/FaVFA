import pandas as pd
import argparse
import numpy as np

from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_ratio, equalized_odds_difference
from statsmodels.formula.api import ols, logit
from statsmodels.stats.anova import anova_lm
from colorama import Fore, Back, Style, init

# Initialize Colorama
init(autoreset=True)

def load(dataset, results_path):
    """
    Load the dataset metadata and the results from the evaluation. Combines them into a single dataframe.
    :param dataset: name of the evaluation dataset
    :param results_path: path to the results csv, with 3 columns 'person_1', 'person_2', 'dist'
    :return: a combined pd.DataFrame
    """
    results = pd.read_csv(results_path)

    assert tuple(results.columns) == ('img_1', 'img_2', 'dist'), "Results csv must have columns 'person_1', 'person_2', 'dist'"

    # embedding distance to angles
    results['dist'] = results['dist'].apply(lambda x: np.arccos(1 - x ** 2 / 2))

    df = pd.read_csv(f'data/{dataset}.csv')
    df = df.merge(results, left_on=['img_1', 'img_2'], right_on=['img_1', 'img_2'])

    return df


def calculate_accuracy(pos, neg, treshold):
    # pos is an array of distances of the same person
    # neg is an array of distances of different people
    # treshold is the treshold to consider a distance as a match

    pos_matches = np.count_nonzero(pos <= treshold) / len(pos)
    neg_matches = np.count_nonzero(neg > treshold) / len(neg)

    return (pos_matches + neg_matches) / 2


def calculate_thresh(pos, neg):
    # pos is an array of distances of the same person
    # neg is an array of distances of different people
    # we want to find the treshold that minimizes the accuracy
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    all_distances = np.hstack((pos, neg))
    best_acc = 0
    best_thresh = 0
    for tresh in all_distances:
        acc = calculate_accuracy(pos, neg, tresh)
        if acc > best_acc:
            best_acc = acc
            best_thresh = tresh
    return best_thresh

def compute_fairness_metrics(df):
    # Print basic metrics

    tpr = df[df['y_true'] == 1]['right'].mean()
    fpr = 1 - df[df['y_true'] == 0]['right'].mean()

    macro_avg_accuracy = df.groupby('segment_1')['right'].mean().mean()

    print('Basic metrics')
    print('='*70)
    print(f'Micro-avg accuracy: {df["right"].mean()*100:.2f}')
    print(f'Macro-avg accuracy: {macro_avg_accuracy * 100:.2f}')
    print(f'True positive rate: {tpr*100:.2f}')
    print(f'False positive rate: {fpr*100:.2f}')
    print('='*70)
    print('\n')


    # print fairness metrics
    dpd = demographic_parity_difference(df['y_true'].values, df['y_pred'].values,
                                        sensitive_features=df['segment_1'])
    dpr = demographic_parity_ratio(df['y_true'], df['y_pred'], sensitive_features=df['segment_1'])
    eod = equalized_odds_difference(df['y_true'], df['y_pred'], sensitive_features=df['segment_1'])
    eor = equalized_odds_ratio(df['y_true'].values, df['y_pred'].values,
                               sensitive_features=df['segment_1'].values)

    print('Fairness metrics')
    print('=' * 70)
    print(f'Demographic parity difference: {dpd*100:.2f}')
    print(f'Demographic parity ratio: {dpr*100:.2f}')
    print(f'Equalized odds difference: {eod*100:.2f}')
    print(f'Equalized odds ratio: {eor*100:.2f}')
    print('=' * 70)
    print('\n')


def anova_table(aov):
    """
    Helper function to compute some ANOVA statistics
    :param aov: an object returned by anova_lm
    :return: pd.DataFrame with the ANOVA statistics
    """
    aov['mean_sq'] = aov[:]['sum_sq']/aov[:]['df']

    aov['eta2'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])

    aov['omega2'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*aov['mean_sq'][-1]))/(sum(aov['sum_sq'])+aov['mean_sq'][-1])

    cols = ['sum_sq', 'df', 'mean_sq', 'F', 'PR(>F)', 'eta2', 'omega2']
    aov = aov[cols]
    return aov

def compute_anova(df, alpha):
    """
    Compute an ANOVA on the data
    :param df: pd.DataFrame of the pair data
    :param alpha: float, significance level
    """
    model = ols(
        f"dist ~ gender + age + ethnicity + angle",
        df)

    res = model.fit()
    p_value = res.f_pvalue
    r = res.rsquared

    aov_table = anova_table(anova_lm(res, typ=1)).sort_values('eta2')[['PR(>F)', 'eta2']].rename(columns={'PR(>F)': 'p_value'})

    if p_value >= alpha:
        print(f'{Fore.RED}ANOVA not significant, p_value={p_value:.3f}')
        return

    print(f'Total explained variance: {r*100:.1f}%, {Fore.GREEN}p_value = {p_value:.3f}')
    for i, row in aov_table.iterrows():
        if i == 'Residual':
            continue
        p_value = row['p_value']
        if p_value >= alpha:
            p_color = Fore.RED
        else:
            p_color = Fore.GREEN
        print(f"{i:<15}explained variance = {row['eta2']*100:.1f}%, {p_color}p_value = {p_value:.3f}")

def get_effect_colors(row, alpha):
    """ Helper function to get the colors for the effect and p_value """
    if row['Pr(>|z|)'] >= alpha:
        p_color = Fore.RED
        effect_color = Fore.YELLOW
    else:
        p_color = Fore.GREEN
        if row['dy/dx'] > 0:
            effect_color = Fore.GREEN
        else:
            effect_color = Fore.RED

    return effect_color, p_color

def compute_marginal_effects(df, dataset, alpha=0.05):
    """
    Compute the marginal effects of the model
    :param df: pd.DataFrame of the pair data
    :param dataset: str, name of the dataset
    :param alpha: float, significance level
    """

    # First select the name of the reference "ethnicity" group
    if dataset in ['favcid', 'bfw']:
        base_ethnicity = 'White x White'
    elif dataset == 'rfw':
        base_ethnicity = 'Caucasian x Caucasian'
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    # Compute the logit regression
    model = logit(
        f'right ~ C(ethnicity, Treatment(reference="{base_ethnicity}")) + C(gender, Treatment(reference="Male x Male")) + age + x_dist + y_dist + z_dist',
        df)
    res = model.fit(maxiter=100, disp=0)

    # Get marginal effects and p-values at the alpha level
    res = res.get_margeff().summary_frame(alpha=alpha).reset_index(names='variable')

    # Separate the different variables
    ethnicity_effects = res[res['variable'].str.contains("ethnicity")]
    gender_effects = res[res['variable'].str.contains("gender")]
    other_effects = res[(~res['variable'].str.contains("gender")) & (~res['variable'].str.contains("ethnicity"))]

    # Print the results
    print(f"Ethnicity Effects, reference = {base_ethnicity}")
    print('-' * 70)
    for i, row in ethnicity_effects.iterrows():
        variable = row['variable'].split('[')[1][2:-1]
        margin = row['Cont. Int. Hi.'] - row['dy/dx']
        effect_color, p_color = get_effect_colors(row, alpha)
        print(f"{variable:<25} : Effect = {effect_color}{row['dy/dx']*100: .1f} ±{margin*100: .1f}, {p_color}p_value={row['Pr(>|z|)']:.3f}")
    print()

    print(f"Gender Effects, reference = Male x Male")
    print('-' * 70)
    for i, row in gender_effects.iterrows():
        variable = row['variable'].split('[')[1][2:-1]
        margin = row['Cont. Int. Hi.'] - row['dy/dx']
        effect_color, p_color = get_effect_colors(row, alpha)
        print(f"{variable:<25} : Effect = {effect_color}{row['dy/dx'] * 100: .1f} ±{margin*100: .1f}, {p_color}p_value={row['Pr(>|z|)']:.3f}")

    print()

    print(f"Continuous Effects")
    print('-' * 70)
    for i, row in other_effects.iterrows():
        variable = row['variable']
        margin = row['Cont. Int. Hi.'] - row['dy/dx']
        effect_color, p_color = get_effect_colors(row, alpha)
        print(f"{variable:<25} : Effect = {effect_color}{row['dy/dx'] * 100: .1f} ±{margin*100: .1f}, {p_color}p_value={row['Pr(>|z|)']:.3f}")


def compute(dataset, results_path, alpha):
    """
    Compute all the results of the paper for a given dataset and a given results file
    :param dataset: str, name of the dataset
    :param results_path: str, path to the results file
    :param alpha: float, significance level
    """
    # Load the data
    df = load(dataset, results_path)

    # Calculate the best threshold to separate the positive and negative examples
    best_tresh = calculate_thresh(df[df['y_true']==1]['dist'].values, df[df['y_true']==0]['dist'].values)

    # Get binary predictions & create a variable for correct prediction
    df['y_pred'] = (df['dist'] <= best_tresh).astype(int)
    df['right'] = (df['y_true'] == df['y_pred']).astype(int)

    # keep only hard pairs
    df = df[df['segment_1'] == df['segment_2']]

    # Print basic metrics and table 2
    compute_fairness_metrics(df)

    # Print ANOVA results
    positive = df[df['y_true'] == 1]
    negative = df[df['y_true'] == 0]
    print('Latent Space Variance Analysis (ANOVA)')
    print('='*70)
    print('Positive pairs')
    print('-'*70)
    compute_anova(positive, alpha)
    print('\n')
    print('Negative pairs')
    print('-'*70)
    compute_anova(negative, alpha)
    print()
    print("Marginal Effects")
    print('='*70)
    print('On TPR')
    print('-'*70)


    # Print marginal effects on TPR and FPR
    compute_marginal_effects(positive, dataset, alpha)
    print()
    print('On FPR')
    print('-'*70)
    negative_ = negative.copy()
    negative_['right'] = 1 - negative_['right']
    compute_marginal_effects(negative_, dataset, alpha)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute fairness metrics for a dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the evaluation dataset.")
    parser.add_argument('--results_path', type=str, required=True, help="Path to the results csv. Must have columns in order 'img_1', 'img_2'', 'dist'")
    parser.add_argument('--alpha', type=float, required=False, default=0.05, help="Significance level for ANOVA and marginal effects")

    args = parser.parse_args()
    dataset = args.dataset
    results_path = args.results_path
    alpha = args.alpha

    assert dataset in ['favcid', 'rfw', 'bfw'], "Dataset must be one of 'favcid', 'rfw', 'bfw'"
    assert 0 < alpha < 1, "Alpha must be between 0 and 1"

    compute(dataset, results_path, alpha)