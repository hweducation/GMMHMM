import pandas as pd
import matplotlib.pyplot as plt
in_dir = 'E://out/sum/hou-shu-01-output/lowHighTypicalCopy.csv'

df = pd.read_csv(in_dir, header=0)

print("原始数据的大小：", df.shape)

df.dropna(subset=['highSupp', 'lowSupp'], inplace = True)
# timestamp = df['Recording timestamp [ms]']
highSupp = df['highSupp']
lowSupp = df['lowSupp']
plt.scatter(highSupp, lowSupp, s=5,marker="o")
plt.show()