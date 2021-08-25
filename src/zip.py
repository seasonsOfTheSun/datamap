import os
import pandas as pd
import numpy as np

def join_files(foldername):
    out = []
    for i in os.listdir(foldername):
        temp = pd.read_csv(foldername +"/"+ i, index_col = 0, header = None)
        temp = temp[1]
        out.append(temp)

    index = set([]).union(*[set(i.index) for i in out])
    temp = [temp.reindex(list(index)) for temp in out]
    df = pd.concat(out, axis = 1).transpose()
    return df

if __name__ == "__main__":
   
   import sys
   foldername = sys.argv[1]
   if foldername[-1] == "/":
       foldername = foldername[:-1]
       
   df = join_files(foldername)
   df.to_csv(foldername + ".csv")
