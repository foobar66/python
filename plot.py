import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/home/philip/Desktop/COVID19BE_CASES_MUNI.csv")
arr = data.to_numpy()
cities_dup = arr[:,1]
cities = list(dict.fromkeys(cities_dup))
cities.sort()
result = list([ (city, sum([ x[2] for x in arr if x[1] == city])) for city in cities ])
result_arr = np.array(result)
xdata = list(result_arr[120:140][:,0])
ydata = [ int(x) for x in list(result_arr[120:140][:,1]) ]

#plt.xticks(range(len(xdata)), xdata,rotation=90)
#plt.ylabel('Cases')
#plt.title('COVID cases per city')
#plt.bar(range(len(ydata)), ydata) 
#plt.show()

dic = dict([ [ str, sum(np.array(data[data['TX_DESCR_NL'] == str])[:,2]) ] for str in set(data['TX_DESCR_NL']) ])


