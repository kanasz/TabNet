import numpy as np
from imblearn.metrics import geometric_mean_score
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from base_finctions import get_slovak_data
import torch
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.version.cuda)
print(torch.device('cuda'))
print(torch.cuda.is_available())
print(DEVICE)

class GMean(Metric):
    def __init__(self):
        self._name = "GMean"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.where(y_score[:, 1] > 0.5, 1, 0)
        gmean = geometric_mean_score(y_true, y_pred)
        return gmean

def do_prediction(sector,year,postfix):
        X, y = get_slovak_data(sector, year, postfix)
        y = y.to_numpy()
        X = SimpleImputer().fit_transform(X)
        kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        gmeans = []
        for train_index, test_index in kf.split(X, y):
                X_train, X_valid = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]
                tb_cls = TabNetClassifier(seed=42,
                                          device_name=DEVICE,
                                          mask_type='sparsemax',
                                          optimizer_fn=torch.optim.Adam,
                                          optimizer_params=dict(lr=2e-2))
                tb_cls.fit(X_train, y_train,
                           eval_metric=[GMean],
                           max_epochs=1000, patience=100,
                           drop_last=False)
                predictions = tb_cls.predict_proba(X_valid)

                y_pred = np.where(predictions[:, 1] > 0.5, 1, 0)
                gmean = geometric_mean_score(y_valid, y_pred)
                gmeans.append(gmean)

        df = pd.DataFrame(columns=['Sector', 'Year', 'Postfix', 'GM_0', 'GM_1', 'GM_2', 'GM_3', 'GM_4'])

        #print([sector, year, postfix] + gmeans)
        s = pd.Series([sector, year, postfix] )
        new_row = {'Sector': sector, 'Year': year, 'Postfix': postfix,'GM_0':gmeans[0],'GM_1':gmeans[1],'GM_2':gmeans[2],
                   'GM_3':gmeans[3], 'GM_4':gmeans[4]}
        df = df.append(new_row, ignore_index=True)
        return df





results = pd.DataFrame()
result = do_prediction('agriculture', 13, '12')
results = results.append(result,ignore_index=True)
print(result)
'''
result = do_prediction('agriculture', 14, '13')
results = results.append(result,ignore_index=True)

result = do_prediction('agriculture', 15, '14')
results = results.append(result,ignore_index=True)

result = do_prediction('agriculture', 16, '15')
results = results.append(result,ignore_index=True)

result = do_prediction('construction', 13, '12')
results = results.append(result,ignore_index=True)

result = do_prediction('construction', 14, '13')
results = results.append(result,ignore_index=True)

result = do_prediction('construction', 15, '14')
results = results.append(result,ignore_index=True)

result = do_prediction('construction', 16, '15')
results = results.append(result,ignore_index=True)

result = do_prediction('manufacture', 13, '12')
results = results.append(result,ignore_index=True)

result = do_prediction('manufacture', 14, '13')
results = results.append(result,ignore_index=True)

result = do_prediction('manufacture', 15, '14')
results = results.append(result,ignore_index=True)

result = do_prediction('manufacture', 16, '15')
results = results.append(result,ignore_index=True)

result = do_prediction('retail', 13, '12')
results = results.append(result,ignore_index=True)

result = do_prediction('retail', 14, '13')
results = results.append(result,ignore_index=True)

result = do_prediction('retail', 15, '14')
results = results.append(result,ignore_index=True)

result = do_prediction('retail', 16, '15')
results = results.append(result,ignore_index=True)
'''
#results.to_csv('results.csv')






