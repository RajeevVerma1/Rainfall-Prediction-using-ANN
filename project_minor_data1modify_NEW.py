from keras.models import Sequential
from keras.layers import Dense,Input
from keras.callbacks import History
import pandas as pd
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics

#######Loading CSV file using Pandas

history = History()
df=pd.read_csv('/home/rajeev/minor_project/data1modify.csv')
df=df.dropna()
a=list(df.columns)
df1=df[[a[0],a[1],a[2],a[3],a[4],a[5]]]
Y = df[a[6]]
print(df)

#######################################  
model = Sequential()

model.add(Dense(20, activation='sigmoid', input_shape=(6,)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
print(model.summary())

X_train, X_test, Y_train, Y_test = train_test_split(df1,Y, test_size=0.2, random_state=42)
history1 = model.fit(X_train, Y_train,validation_data=(X_test,Y_test),epochs=100,batch_size=100,verbose=2, callbacks = [history])
print(X_test)
print(Y_test)
predictions = model.predict(X_test)
print(predictions)

model.save('/home/rajeev/minor_project/model_trial.h5')
print("Saved model to disk")

plt.plot(history1.history['mean_squared_error'])
plt.plot(history1.history['val_mean_squared_error'])

plt.ylabel('Mean Squared Error')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()
