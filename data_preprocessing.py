from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def load_dataset():
    df= pd.read_csv("Student_Depression_Dataset.csv")
    df["Depression"]= df["Depression"].map({"Yes":1,"No":0})
    df= df.drop(columns=["Home_division"])
    return df


def preprocess_data(df):
    ordinal_cat_columns= ["Health_condition","Marital_status","Family_background","Living_style","friend_circle"]
    nonordinal_cat_columns= ["Gender","Relationship_status"]
    num_normalization_columns= ["Age","Semester","CGPA"]

    c_ordinal= Pipeline(
        steps=[
            (
                "ordinal encoder",
                OrdinalEncoder(categories=[
                        ["Critical","Abnormal","Normal","Good"],
                        ["Separated","Unmarried","Married"],
                        ["Lower Middle class","Middle class","Rich class"],
                        ["Bachelor","Family"],
                        ["No","Yes"]
                    ]
                )
            )
        ]
    )
 
    c_nonordinal= Pipeline([("non-ordinal encoder",OneHotEncoder(drop= "first"))])

    num_standarization= Pipeline([("standardization",StandardScaler())])

    num_normalization= Pipeline([("normalization",MinMaxScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("Ordinal encoder",c_ordinal,ordinal_cat_columns),
            ("One hot encoder",c_nonordinal,nonordinal_cat_columns),
            ("Standarizer",num_standarization,["Cost_without_study"]),
            ("Normalizer",num_normalization,num_normalization_columns)
        ],
        remainder="passthrough"
    )

    return preprocessor