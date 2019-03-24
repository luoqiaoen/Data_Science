import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('bank_note_data.csv')

scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])

X = df_feat
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')
feat_cols = [image_var,image_skew,image_curt,entropy]

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=20,shuffle=True)

classifier.train(input_fn=input_func,steps=500)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)

note_predictions = list(classifier.predict(input_fn=pred_fn))

final_preds  = []
for pred in note_predictions:
    final_preds.append(pred['class_ids'][0])

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,final_preds))

print(classification_report(y_test,final_preds))
