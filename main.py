import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential

df=pd.read_csv("airpollution.csv")
# print(df.head())
# print(df.columns)
sns.boxplot(data=df)
# plt.show()

sns.pairplot(data=df)
# plt.show()
print(df.shape)
q1=df["PM2.5 AQI Value"].quantile(0.25)
q3=df["PM2.5 AQI Value"].quantile(0.75)

iqr=q3-q1
min=q1-(1.5*iqr)
max=q3+(1.5*iqr)
filtered_data = df[(df["PM2.5 AQI Value"] >= min) & (df["PM2.5 AQI Value"] <= max)]
print(filtered_data.shape)

# print(filtered_data.head(10))
# print(filtered_data.columns)

filtered_data = filtered_data.dropna(subset=[
    "AQI Value","CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value",
    "Country","City",
    "AQI Category","CO AQI Category","Ozone AQI Category","NO2 AQI Category",
    "PM2.5 AQI Category"
]).reset_index(drop=True)

# print(filtered_data.shape)
x=filtered_data[["CO AQI Value","Ozone AQI Value","NO2 AQI Value","PM2.5 AQI Value"]]

le=LabelEncoder()
y=le.fit_transform(filtered_data["AQI Category"])

# ohe=OneHotEncoder(sparse_output=False,drop="first")
# cat_data=filtered_data[["CO AQI Category","Ozone AQI Category","NO2 AQI Category"]]
# en_data=ohe.fit_transform(cat_data)
# en_dataframe=pd.DataFrame(en_data, columns=ohe.get_feature_names_out(cat_data.columns))

# x_en=pd.concat([x,en_dataframe],axis=1)

# od=OrdinalEncoder()
# od_cat_data=filtered_data[["Country","City"]]
# od_en_data=od.fit_transform(od_cat_data)
# en_od=pd.DataFrame(od_en_data,columns=od_cat_data.columns)

# x_final=pd.concat([x_en,en_od],axis=1)

ss=StandardScaler()
X_final=pd.DataFrame(data=ss.fit_transform(x),columns=x.columns)

x_train,x_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=42)

ann=Sequential()

ann.add(Dense(11,input_dim=4,activation=tf.keras.activations.relu))
ann.add(Dense(10,activation=tf.keras.activations.relu))
ann.add(Dense(8,activation=tf.keras.activations.relu))
ann.add(Dense(7,activation=tf.keras.activations.relu))
ann.add(Dense(6,activation=tf.keras.activations.softmax))

ann.compile(optimizer="adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history=ann.fit(x_train,y_train,batch_size=150,epochs=10)

print("Train Accuracy for all the epochs ======================================")
print(history.history["accuracy"])
print("Final Train Accuracy")
print(history.history["accuracy"][-1])

test_loss,test_accuracy=ann.evaluate(x_test,y_test)
print("Test Accuracy=================")
print(test_accuracy)