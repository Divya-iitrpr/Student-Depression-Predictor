from data_preprocessing import load_dataset,preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


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