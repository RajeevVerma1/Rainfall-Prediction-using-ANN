from keras.models import load_model
import pandas as pd


loaded_model=load_model('/home/rajeev/minor_project/model_trial.h5')

x1=int(input('Temperature(C):'))
x2=int(input('Pressure(mB):'))
x3=int(input('Humidity(%):'))
x4=int(input('Cloud cover(%):'))
x5=int(input('Wind speed(mph):'))
x6=int(input('Wind Direction:'))

input_value = [{'temp(c)': x1, 'pressure(mb)': x2, 'humidity()': x3, 'cloud cover()': x4, 'wind speed(mph)': x5, 'wind dir.':x6}]
X_test = pd.DataFrame(input_value)

predictions = loaded_model.predict(X_test)
if(predictions[0]<0):
    predictions[0]=0
#print(predictions)
print("Precipitations(mm):",predictions[0])

