import pandas as pd
import numbers
import numpy as np



def get_cols(df):

    name=df[:-4]
    data=pd.read_csv(df)
    cols=data.columns
    pos=[]
    neg=[]

    for col in cols:
        for item in data[col]:
            if isinstance(item,numbers.Number):

                if item>0:
                    item=round(item,3)
                    pos.append(item)
                elif item<0:
                    item = round(item, 3)
                    neg.append(item)



    pos=np.asarray(pos)
    neg=np.asarray(neg)

    print("pos shape is ",pos.shape)
    print("neg shape is ", neg.shape)

    if len(pos)>=len(neg):
        l=len(pos)-len(neg)
        a = np.empty(l)
        a[:] = np.nan
        neg=np.concatenate((neg,a),axis=None)

    else:
        l = len(neg) - len(pos)
        a = np.empty(l)
        a[:] = np.nan
        pos=np.concatenate((pos, a), axis=None)



    return_df=pd.DataFrame()

    return_df[name +" positive"]= pos
    return_df[name + " negative"] = neg

    return_df.to_csv(name +'processed.csv')

    return return_df






