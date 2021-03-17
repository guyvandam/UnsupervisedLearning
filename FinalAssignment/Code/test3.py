import pandas as pd

df = pd.DataFrame()
data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]

data1 = [1,2,3]
data2 = [3,4,5]
data3 = [0,0,0]
df['col1'] = data1
df['col2'] = data2
df['col3'] = data3

df.loc['mean'] = df.mean()
print(df)
s = df.T['mean'].pct_change().T * 100
df.loc['pct_change'] = s
print(df)