
import pandas
import numpy
import sys


dataset = sys.argv[1]
metadata = pandas.read_csv(f"data/intermediate/{dataset}/metadata.csv", index_col = 0)


known = metadata[metadata.known == 1].index
test = numpy.random.choice(known, len(known)//3, replace = False)
train = [i for i in known if i not in test]


pandas.Series(train).to_csv(f"data/intermediate/{dataset}/train.csv", index=None, header = None)
pandas.Series(test).to_csv(f"data/intermediate/{dataset}/test.csv",   index=None, header = None)

