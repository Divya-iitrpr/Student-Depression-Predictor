from data_preprocessing import load_dataset,preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, ConfusionMatrixDisplay, RocCurveDisplay, auc
import matplotlib.pyplot as plt

print("="*90)
print("[AI101] PROJECT: STUDENT DEPRESSION PREDICTION USING LOGISTIC REGRESSION")
print("="*90)


print("[-] DATASET LOADING...")
df= load_dataset()
print("[✓] Dataset Loaded Successfully!\n")
print(f"Shape: {df.shape}")
print()

# train_test splitting
X= df.drop(columns=["Depression"])
y= df["Depression"]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.20,random_state=19,stratify=y)

print("[-] DATA PREPROCESSING...")
preprocessor= preprocess_data(df)
X_train= preprocessor.fit_transform(X_train)
X_test= preprocessor.transform(X_test)

print("[✓] Data preprocessed successfully!\n")
print("Data training samples: ", X_train.shape[0])
print("Data testing samples: ", X_test.shape[0])
print("No. of Feature Columns After preprocessing: ",X_train.shape[1])
print()

print("[-] MODEL TRAINING...")
model= LogisticRegression()
model.fit(X_train,y_train)
print("[✓] Model trained Successfully!\n")
print(f"Model WEIGHTS: \nIntercept: {round(model.intercept_[0],3)}")
print("Coefficients:",model.coef_[0].round(3))
print()

def print_results(y_predict, y_true):
    print(f"ACCURACY : {100*accuracy_score(y_predict, y_true):.3f} %")
    print(f"PRECISION : {precision_score(y_predict, y_true):.3f}")
    print(f"RECALL : {recall_score(y_predict, y_true):.3f}")
    print(f"F1_SCORE : {f1_score(y_predict, y_true):.3f}")

print("[-] MODEL EVALUATION : TRAINING DATA")
print_results(model.predict(X_train),y_train)
print()

print("[-] MODEL EVALUATION : TEST DATA")
print_results(model.predict(X_test),y_test)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test,model.predict(X_test)), display_labels=["No Depression","Depression"])
disp.plot(cmap="Blues")
plt.savefig("outputs/cm.png")
print("[✓] Saved Confusion matrix in outputs!\n")

fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
roc_auc = auc(fpr, tpr)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()
plt.savefig("outputs/roc.png")
print("[✓] Saved ROC Curve in outputs!\n")
