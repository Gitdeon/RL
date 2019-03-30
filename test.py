stats = {'a':1000, 'b':3000, 'c': 100}
a=max(stats, key=stats.get)
print(a)