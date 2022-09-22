##############
#%% import %%#
##############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#################
#%% variables %%#
#################

# path
DATA_PATH = os.path.join(os.getcwd(), 'data', 'data.csv')

# random seed
SEED = 42

####################
#%% data loading %%#
####################
df = pd.read_csv(DATA_PATH)
df


#######################
#%% data inspection %%#
#######################
df.info()

df.describe().T

df.isna().mean()

df.duplicated().sum() # 1 duplicated
df.drop_duplicates(inplace=True)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import set_config
set_config(display="diagram")

#############
#%% split %%#
#############

# feature
X = df.drop(columns=['output'])

# target
y = df['output']

##################
#%% train test %%#
##################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)


#########################
#%% Model Development %%#
#########################
model = Pipeline(
    [
        ('s_scaler', StandardScaler()),
        ('rf', RandomForestClassifier())
    ]
)

# fit and validate
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = model.score(X_test, y_test)
# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Mean accuracy: %.2f%%\n" % score)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, RocCurveDisplay

##################
#%% Evaluation %%#
##################
report = classification_report(y_test, y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()
report.to_json('metrics.json')

cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=80)

y_score = model.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score[:,1], pos_label=model.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.title("ROC Curve")
plt.savefig('ROC Curve.png', dpi=80)
