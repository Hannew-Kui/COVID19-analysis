import pandas as pd
import json
import matplotlib.pyplot as plt



def drawPlot(df,title=""):
    # 绘制折线图  
    plt.plot(df.index, df['value'])  # marker='o' 表示在每个数据点上画一个圆圈  

    # 设置x轴和y轴的标签  
    plt.xlabel('Time')  
    plt.ylabel('Value')  

    # 设置图的标题  
    plt.title(title)  

    # 显示图像  
    plt.show()  

    # 如果你想要保存图像到文件  
    plt.savefig(title+'.png')
    

# data_df=pd.read_csv("./CSSEcovid19/time_series_covid19_confirmed_global.csv")

# dates=data_df.columns[4:]

# print(dates)

# hubei=data_df.loc[73]
# Y=[hubei[dates[0]]]
# for day in range(1,len(dates)):
#     Y.append(hubei[dates[day]]-hubei[dates[day-1]])
# print(Y)
# drawPlot(pd.DataFrame(Y,index=dates,columns=['value']))




