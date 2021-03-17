from scipy.stats import f_oneway
import pandas as pd
from functools import cmp_to_key
from scipy.stats import ttest_ind

# test_results_df = pd.DataFrame(columns=['p_value', 'stat', 'test', 'result'])
test_results_df = pd.DataFrame(columns=['p_value', 'stat', 'test', 'result'])

def add_results_to_df(result, p_value, stat, test):
    global test_results_df
    row = pd.Series([p_value, stat, test, result], index=test_results_df.columns)
    test_results_df = test_results_df.append(row, ignore_index = True)

def compare(item1, item2):

    k1 = item1[0]
    k2 = item2[0]

    item1 = list(item1[1].values())
    item2 = list(item2[1].values())

    stat, p_value = ttest_ind(item1, item2) # same mean.
    test = f"{k1} = {k2}"
    test_succes = p_value > 0.05
    if p_value > 0.05:
        # result = f"{k1} probably have the same mean as {k2}"
        # add_results_to_df(result, p_value, stat, test)
        add_results_to_df(test_succes, p_value, stat, test)
        return 0
    else:
        # result = f"{k1} probably dosen't have the same mean as {k2}"
        # add_results_to_df(result, p_value, stat, test)
        add_results_to_df(test_succes, p_value, stat, test)

        test = f"{k2} > {k1}"
        test = f"{k1} > {k2}"
        stat, p_value = ttest_ind(item1, item2, alternative='greater') # item1 is bigger than item 2.
        test_succes = p_value > 0.05
        if p_value > 0.05:
            # result = f"{k1} probably have bigger mean than {k2}"
            # add_results_to_df(result, p_value, stat, test)
            add_results_to_df(test_succes, p_value, stat, test)
            return 1
        else:
            # result = f"{k2} probably have bigger mean than {k1}"
            # add_results_to_df(result, p_value, stat, test)
            add_results_to_df(test_succes, p_value, stat, test)
            return -1

def run_anova(df):
    value_list_list = df.T.values.tolist()
    stat, p_value = f_oneway(*value_list_list)

    columns_list = list(df.columns)
    columns_list = [str(x) for x in columns_list]
    # columns_string = ', '.join(columns_list)

    test_string = ' = '.join(columns_list)
    test_succes = p_value > 0.05
    add_results_to_df(test_succes, p_value, stat, test_string)

    # if p_value > 0.05:
        # result = f"{columns_string} probably have the same distribution"
        # add_results_to_df(result, p_value, stat, test_string)

    # else:
        # result = f"{columns_string} probably have different distributions"
        # add_results_to_df(result, p_value, stat, test_string)
    
    return stat, p_value

def sort_df_by_stat_test(df):
    global test_results_df
    test_results_df = pd.DataFrame(columns=['p_value', 'stat', 'test', 'result'])

    sorted_df = df.copy()
    _ , p_value = run_anova(df)

    if p_value < 0.05:
        dict_list = df.to_dict()
        sorted_dict = dict(sorted(dict_list.items(), key=cmp_to_key(compare)))
        sorted_df = pd.DataFrame(sorted_dict)
    
    # print("sorted_df \n", sorted_df)
    # print("test results \n", test_results_df)

    return test_results_df, sorted_df
if __name__ == '__main__':
    df = pd.DataFrame()
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [1.142, -0.432, -0.938, -0.729, -0.846, -0.157, 0.500, 1.183, -1.075, -0.169]
    data3 = [-0.208, 0.696, 0.928, -1.148, -0.213, 0.229, 0.137, 0.269, -0.870, -1.204]

    # data1 = [1,2,3]
    # data2 = [3,4,5]
    # data3 = [0,0,0]

    df['col1'] = data1
    df['col2'] = data2
    df['col3'] = data3

    sort_df_by_stat_test(df)