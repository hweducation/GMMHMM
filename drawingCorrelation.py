# import numpy as np
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import cm, pyplot as plt
in_dir = 'E://out/sum/hou-shu-01-output/lowHighTypicalCopy.csv'

print("in_dir")
print(in_dir)
df = pd.read_csv(in_dir, header=0)

print("原始数据的大小：", df.shape)
#print("原始数据的列名", df.columns)

df.dropna(subset=['highSupp', 'lowSupp'], inplace = True)
# timestamp = df['Recording timestamp [ms]']
highSupp = df['highSupp']
lowSupp = df['lowSupp']

plt.scatter(highSupp, lowSupp, s=10)
plt.show()
#复制到命令行跑