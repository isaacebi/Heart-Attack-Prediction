import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

import pandas as pd
from lightgbm import LGBMClassifier
from mlem.api import save
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.load_params import load_params
from xgboost import XGBClassifier


def train(data_dir,
          model_type,
          random_state,
          **train_params):
    X_train = pd.read_pickle(data_dir/'X_train.pkl')
    y_train = pd.read_pickle(data_dir/'y_train.pkl')
    
    if model_type == "randomforest":
        clf = RandomForestClassifier(random_state=random_state, 
                                 **train_params)
    elif model_type == "lightgbm":
        clf = LGBMClassifier(random_state=random_state, 
                                    **train_params)
    elif model_type == "xgboost":
        clf = XGBClassifier(random_state=random_state, 
                                    **train_params)

    model = Pipeline(
        [
            ('s_scaler', StandardScaler()),
            ('clf', clf)
        ]
    )
    
    model.fit(X_train, y_train)
    save(
        model,
        "clf-model",
        sample_data=X_train
    )

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    params = load_params(params_path=args.config)
    data_dir = Path(params.base.data_dir)
    random_state = params.base.random_state
    model_type = params.train.model_type
    train_params = params.train.params
    train(data_dir=data_dir,
          model_type=model_type,
          random_state=random_state,
          **train_params)