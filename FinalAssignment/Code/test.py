# global test_results_df
# Example of the Analysis of Variance Test
from scipy.stats import f_oneway
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
# print(df)
from functools import cmp_to_key

test_results_df = pd.DataFrame(columns=['p_value', 'stat', 'result'])
def add_results_to_df(result, p_value, stat):
    global test_results_df
    row = pd.Series([p_value, stat, result], index=test_results_df.columns)
    test_results_df = test_results_df.append(row, ignore_index = True)
    print(result)

def compare(item1, item2):
    from scipy.stats import ttest_ind

    k1 = item1[0]
    k2 = item2[0]

    item1 = list(item1[1].values())
    item2 = list(item2[1].values())

    stat, p_value = ttest_ind(item1, item2) # same mean.
    
    if p_value > 0.05:
        result = f"{k1} probably have the same mean as {k2}"
        add_results_to_df(result, p_value, stat)
        return 0
    else:
        result = f"{k1} probably dosen't have the same mean as {k2}"
        add_results_to_df(result, p_value, stat)

        stat, p_value = ttest_ind(item1, item2, alternative='greater') # item1 is bigger than item 2.
        if p_value > 0.05:
            result = f"{k1} probably have bigger mean than {k2}"
            add_results_to_df(result, p_value, stat)
            return 1
        else:
            result = f"{k2} probably have bigger mean than {k1}"
            add_results_to_df(result, p_value, stat)
            return -1

# Calling
dict_list = df.to_dict()
# l = sorted(list_list, cmp = compare)
l = dict(sorted(dict_list.items(), key=cmp_to_key(compare)))
df = pd.DataFrame(l)
print(df)
print(test_results_df)

def sort_df_by_stat_test(df):

    stats , p_value = run_anova(value_list_list)

    if p_value > 0.05:
        print('Probably the same distribution')
        return stats, p_value

