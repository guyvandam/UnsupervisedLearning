import pandas as pd
from scipy import stats
df = pd.DataFrame({'y': [4,4,3]})
print( df)
pop_mean = float(df.mean())
print(float(pop_mean))
p, stat = stats.ttest_1samp(df, pop_mean)

print(p)