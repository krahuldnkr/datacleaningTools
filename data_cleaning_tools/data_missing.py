import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# to think:: How to handle missing data in case of strings
class dataMissingPreprocess:

    def __init__(self, data_input: dict):
        """_summary_

        Args:
            data_input (dict): _description_
        """
        self.data = (data_input)
        self.null_data = False
        self.null_cols = []

    def isNullValuePresent(self):
        """_summary_
        """
        missingSummary = self.data.isnull().sum()
        print("Following is the data missing report: ")
        print(missingSummary)

        # check if data missing
        for key, row in missingSummary.items():
            if(row > 0):
                self.null_data = True
                self.null_cols.append(key)
        
        print("flag for missing data")
        print(self.null_data)
        print(self.null_cols)
        
    def removeMissingRows(self):
        """_summary_
        """
        if self.null_data:
            data_cleaned = self.data.dropna()

            print("after dropping missing values")
            print(data_cleaned)
            print(self.data)

        else:
            print("invalid operation, no missing data!!")

    def basicImputation(self):
        """_summary_
        """
        df_imputed = self.data.copy()

        print()
        for cols in self.null_cols:
            print("debug")
            print(self.data[cols])
            try:
                df_imputed[cols] = self.data[cols].fillna(self.data[cols].mean())
            except:
                print(self.data[cols].mode())
                df_imputed[cols] = self.data[cols].fillna(self.data[cols].mode()[0])

        print("simple imputation using mean results")
        print(df_imputed)

    def groupedImputation(self, attr1: str, attr2: str, operation: str):
        """_summary_

        Args:
            attr1 (str): _description_
            attr2 (str): _description_
            operation (str): _description_
        """
        df_imputed_by_grp = self.data.copy()
        
        for cols in self.null_cols:
            
            df_imputed_by_grp[cols] = self.data[cols].fillna(self.data.groupby([attr1, attr2])[cols].transform(operation))

        print("group based imputation")
        print(df_imputed_by_grp)

    def regressionBasedImputation(self, dependent_var: list, independent_var : str):
        
        # predicted data
        frame_predicted = self.data.copy()
        print("frame before prediction")
        print(frame_predicted)

        # create split in the data set
        train = self.data[self.data[independent_var].notnull()]
        test = self.data[self.data[independent_var].isnull()]

        X_train = train[dependent_var]
        Y_train = train[independent_var]

        model = LinearRegression().fit(X_train, Y_train)
        frame_predicted.loc[df[independent_var].isnull(), independent_var] = model.predict(test[dependent_var])

        print("predicted frame")
        print(frame_predicted)

    def missingnessAsFeature(self):
        """_summary_
        """
        self.data['Income_missing'] = self.data['Income'].isnull().astype(int)
        self.data['Credit_missing'] = self.data['CreditScore'].isnull().astype(int)

        # Use groupby or crosstab
        print(pd.crosstab(self.data['Defaulted'], self.data['Income_missing']))
        print(pd.crosstab(self.data['Defaulted'], self.data['Credit_missing']))

## usage
# creating a healthcare dataset
# df = pd.DataFrame({
#     'PatientID' : [1, 2, 3, 4, 5],
#     'Age' : [25, 32, 45, 60, 20],
#     'Gender' : ['M', 'F', np.nan, 'M', 'F'],
#     'Diagnosis' : ['A', 'B', 'A', 'A', 'B'],    
#     'Height': [5.5, 6.1, np.nan, 5.9, 6.2],
#     'Weight': [150, 180, 165, 170, 190]
# })  

# problem:: identify inconsistent missingness patters
# analyze whether missingness in Income or creditScore is correlated with defaulted status
df = pd.DataFrame({
    'LoanID': [101, 102, 103, 104, 105],
    'Income': [50000, np.nan, 62000, np.nan, 58000],
    'CreditScore': [700, 680, np.nan, 710, np.nan],
    'Defaulted': [0, 1, 0, 1, 1]
})


objHandle = dataMissingPreprocess(df)
objHandle.isNullValuePresent()
objHandle.removeMissingRows()
objHandle.basicImputation()
#objHandle.groupedImputation('Gender','Diagnosis','mean')
# dependentVar   = [] 
# independentVar = []
# objHandle.regressionBasedImputation()
objHandle.missingnessAsFeature()