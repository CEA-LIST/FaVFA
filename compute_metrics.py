import pandas as pd
import argparse
import numpy as np

from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_ratio, equalized_odds_difference

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
    print('Basic metrics')
    print('='*50)
    print(f'Micro-avg accuracy: {df["right"].mean()*100:.2f}')
    print(f'Standard deviation: {df["right"].std()*100:.2f}')
    print('='*50)
    print('\n')

    # get macro-avg-accuracy
    macro_avg_accuracy = df.groupby('segment_1')['right'].mean().mean()

    # print fairness metrics
    dpd = demographic_parity_difference(df['y_true'].values, df['y_pred'].values,
                                        sensitive_features=df['segment_1'])
    dpr = demographic_parity_ratio(df['y_true'], df['y_pred'], sensitive_features=df['segment_1'])
    eod = equalized_odds_difference(df['y_true'], df['y_pred'], sensitive_features=df['segment_1'])
    eor = equalized_odds_ratio(df['y_true'].values, df['y_pred'].values,
                               sensitive_features=df['segment_1'].values)

    print('Fairness metrics')
    print('=' * 50)
    print(f'Macro-avg accuracy: {macro_avg_accuracy * 100:.2f}')
    print(f'Demographic parity difference: {dpd*100:.2f}')
    print(f'Demographic parity ratio: {dpr*100:.2f}')
    print(f'Equalized odds difference: {eod*100:.2f}')
    print(f'Equalized odds ratio: {eor*100:.2f}')
    print('=' * 50)
    print('\n')



def compute_anova(df):
    pass

def compute_marginal_effects(df):
    pass


def compute(dataset, results_path):
    df = load(dataset, results_path)

    # Calculate the best threshold to separate the positive and negative examples
    best_tresh = calculate_thresh(df[df['y_true']==1]['dist'].values, df[df['y_true']==0]['dist'].values)

    # Get binary predictions
    df['y_pred'] = (df['dist'] <= best_tresh).astype(int)

    # Create a variable for correct prediction
    df['right'] = (df['y_true'] == df['y_pred']).astype(int)

    # keep only hard pairs
    df = df[df['segment_1'] == df['segment_2']]

    compute_fairness_metrics(df)
    compute_anova(df)
    compute_marginal_effects(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute fairness metrics for a dataset.")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the evaluation dataset.")
    parser.add_argument('--results_path', type=str, required=True, help="Path to the results csv. Must have columns in order 'img_1', 'img_2'', 'dist'")


    args = parser.parse_args()
    dataset = args.dataset
    results_path = args.results_path

    assert dataset in ['favcid', 'rfw', 'bfw'], "Dataset must be one of 'favcid', 'rfw', 'bfw'"
    compute(dataset, results_path)