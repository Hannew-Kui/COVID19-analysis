import pandas as pd
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--input_path','-i',type=str,help="input file path")
parser.add_argument('--acc_path','-ap',type=str,help="accumulate data file path")
parser.add_argument("--daily_path",'-dp',type=str,help="daily data file path")




if __name__=='__main__':
    args=parser.parse_args()
    #数据筛选
    with open(args.input_path,"r") as f1:
        df=pd.read_csv(f1)
       
    #数据筛选
    df=df[df['Country/Region'].isin(["China"])]
    df=df[~df['Province/State'].isin(['Unknown'])]
    print("缺失值个数：{}".format(df.isnull().sum()))          #检查缺失值
    df.drop(['Country/Region',"Lat","Long"],axis=1,inplace=True)
    #格式转换
    df=df.T
    columns=df.columns
    new_column={}
    for c in columns:
        new_column[c]=df.iloc[0][c]
    df.rename(columns=new_column, inplace=True)  
    df=df[1:]
    for i in range(1,len(df)):
        for j in range(len(df.iloc[i])):
            if df.iloc[i][j]<df.iloc[i-1][j]:
                df.iloc[i][j]=df.iloc[i-1][j]
    #保存中国各省份的累计数据
    with open(args.acc_path,"w") as f2:
        df.to_csv(f2,index=False)
    #计算并保存每日新增数据
    columns=df.columns
    df=df.diff().iloc[1:]
        #平滑处理
    df=df.rolling(window=5).mean()
    df=df.dropna(axis=0)
    with open(args.daily_path,'w') as f:
        df.to_csv(f,index=False)