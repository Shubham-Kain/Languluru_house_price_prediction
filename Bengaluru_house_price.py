import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
import pickle



df = pd.read_csv(r"C:\Users\hp\.vscode\Project_Machine-Learning\Project-12\Bengaluru_House_Data.csv")

df.drop(columns=["area_type","availability","society","balcony"],inplace=True)
print(df.shape)
# print(df.info())

# for i in df.columns:
#     print(df[i].isnull().sum())
#     print("*"*20)
df["location"] = df["location"].fillna("Sarjapur Road")
df["bhk"] = df["size"].fillna("2 BHK")
df["bath"] = df["bath"].fillna(df["bath"].median())
df["bhk"] = df["bhk"].str.split(" ").str[0].astype(int)
df = df[df["bhk"]<11]
df=df[df["bath"]<10]
df = df[df["price"]<1500]

def convergence(x):

    temp = x.split('-')
    if len(temp)==2:
        return (float(temp[0])+float(temp[1]))/2
    try:
        return float(x)
    except:
        return None
    

df["total_sqft"] = df["total_sqft"].apply(convergence)
df["Prie_per_sqrf"] = np.round(df["price"]*100000/df["total_sqft"])

loc = df["location"].value_counts()
location_less = loc[loc<=10]
# print(loc)
df["location"] = df["location"].apply(lambda x: "other" if x in location_less else x)
# print(df.head(5))
df = df[((df["total_sqft"]/df["bhk"])>=300)]
# print(df.sample(5))
# print(df["price"])
max = df["Prie_per_sqrf"].mean()+(3*df["Prie_per_sqrf"].std())
min = df["Prie_per_sqrf"].mean()-(3*df["Prie_per_sqrf"].std())
df = df[df["Prie_per_sqrf"]<=max]


df.drop(columns=["Prie_per_sqrf","size"],inplace=True)
# print(df.sample(5))
# print(df.shape)

# df.to_csv("Clean_Bengaluru_price.csv")

x = df.drop(columns=["price"])
y = df["price"]


oe = OneHotEncoder()
oe.fit(x[['location']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=3)
col_trans = make_column_transformer((OneHotEncoder(drop="first",categories=oe.categories_,sparse_output=False),['location']),remainder="passthrough")

scaler  = StandardScaler()

xgr = XGBRegressor(n_estimators=200,
                     learning_rate=0.1,
                     max_depth=5,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     gamma=0.1,
                     random_state=42)

pipe = make_pipeline(col_trans,scaler,xgr)
pipe.fit(x_train,y_train)
y_pred = pipe.predict(x_test)
print(r2_score(y_test,y_pred))

pickle.dump(pipe,open("XGBRegressor_pipe.pkl","wb"))



# score = []
# for i in range(1,100):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
#     pipe = make_pipeline(col_trans,scaler,xgr)
#     pipe.fit(x_train,y_train)
#     y_pred_lr = pipe.predict(x_test)
#     score.append(r2_score(y_test,y_pred_lr))

# print(np.argmax(score),score[np.argmax(score)])  


# oe = OneHotEncoder()
# oe.fit(x,y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

# col_trans = make_column_transformer((OneHotEncoder(sparse_output=False,categories=oe.categories_,drop="first"),["location"]),remainder="passthrough")
# scaler = StandardScaler()

# lr = LinearRegression()
# print(df.groupby())

# pipe = make_pipeline(col_trans,scaler,lr)
# pipe.fit(x_train,y_train)
# y_pred_lr = pipe.predict(x_test)
# print(r2_score(y_test,y_pred_lr))
# # sco=[]
# # for i in range(1,100): 
# #    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)  
# #    pipe = make_pipeline(col_trans,scaler,lr)
# #    pipe.fit(x_train,y_train)
# #    y_pred_lr = pipe.predict(x_test,["location","total_sqft","bath","bhk"])
# #    sco.append(r2_score(y_test,y_pred_lr))

# # print(sco)  
