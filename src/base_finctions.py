from pathlib import Path
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from imblearn.metrics import geometric_mean_score, sensitivity_score, specificity_score

def get_slovak_data(business_area, year, postfix):
    print("Loading Slovak data...")
    path_bankrupt = Path(__file__).parent / "data/slovak_data/parsed_data/bankrupt/bankrupt_{}_{}_year_{}.csv" \
        .format(business_area, year, postfix)
    path_non_bankrupt = Path(__file__).parent / "data/slovak_data/parsed_data/non_bankrupt/nonbankrupt_{}_{}_year_{}" \
                                                ".csv".format(business_area, year, postfix)
    print("Data: {}".format(path_bankrupt))
    bankrupt_data = pd.read_csv(path_bankrupt)
    non_bankrupt_data = pd.read_csv(path_non_bankrupt)
    features = bankrupt_data.drop(["IS_BANKRUPT"], axis=1).append(non_bankrupt_data.drop(["IS_BANKRUPT"], axis=1))
    labels = bankrupt_data["IS_BANKRUPT"].append(non_bankrupt_data["IS_BANKRUPT"])
    print("Info: rows - {}, columns - {}".format(len(features), len(features.columns)))
    return features, labels


def get_scoring_dict():
    scoring_dict = {
        'accuracy_score': make_scorer(accuracy_score),
        'f1_score': make_scorer(f1_score),
        'roc_auc_score': make_scorer(roc_auc_score),
        'geometric_mean_score': make_scorer(geometric_mean_score),
        'sensitivity_score': make_scorer(sensitivity_score),
        'specificity_score': make_scorer(specificity_score)
    }
    return scoring_dict