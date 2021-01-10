from scipy.stats import f_oneway


d = {1: [1, 2], 2: [1, 3], 3: [1, 2]}

stat, p = f_oneway(*d.values())

print(p)
