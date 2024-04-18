import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel(r"E:\23年交科\HighwayEnv-master\scripts\highway_ppo\videos_3\drop3_0.xlsx", sheet_name='Sheet1')

data0 = data[data['order']==0]
data1 = data[data['order']==1]
data2 = data[data['order']==2]

x = range(len(data0))
print(data1)
def cauculate_dis(x,y):
    l = []
    for i in range(len(x)):
        l.append(list(x)[i]-list(y)[i]-5)
    return l


# plt.plot(x, data0['x'], label='veh_0')
# plt.plot(x, cauculate_dis(data1['x'],data0['x']), label='2-3dis')
# plt.plot(x, cauculate_dis(data2['x'],data1['x']), label='1-2dis')

plt.plot(x,data0['speed'],label='veh_0')
plt.plot(x,data1['speed'],label='veh_1')
plt.plot(x,data2['speed'],label='veh_2',alpha=0.2)
plt.legend()
plt.grid()
plt.title('speed = self.old_target_speeds[1] + random.gauss(0, 5) * self.dt')
plt.show()
