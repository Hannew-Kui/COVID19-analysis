
import torch  # 导入PyTorch库，用于构建和训练深度学习模型
import os  # 导入os库，用于与操作系统交互
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和分析
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from sklearn.preprocessing import MinMaxScaler  # 导入MinMaxScaler，用于数据标准化
from pandas.plotting import register_matplotlib_converters  # 导入register_matplotlib_converters，用于处理时间序列数据
from torch import nn, optim  # 导入神经网络和优化器模块
import argparse


#设置命令行参数
parser=argparse.ArgumentParser()
parser.add_argument("--mode",'-m',default="test",help="training model or testing model, or use model to predict",choices=['test','train',"predict"])
parser.add_argument('--input_path','-i',default="/data/",help="input file path")
parser.add_argument('--checkpoint_path','-c',help="checkpoint path")
parser.add_argument('--test_output','-t',default="/data/",help="test output path")
parser.add_argument('--num_epochs','-n',type=int,default=300,help="num of epochs")
parser.add_argument('--learning_rate','-lr',default=1e-3,type=float,help="learning rate")    
parser.add_argument('--place','-p',type=str,choices=['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong',
       'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan',
       'Hong_Kong', 'Hubei', 'Hunan', 'Inner Mongolia', 'Jiangsu', 'Jiangxi',
       'Jilin', 'Liaoning', 'Macau', 'Ningxia', 'Qinghai', 'Shaanxi',
       'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Tibet',
       'Xinjiang', 'Yunnan', 'Zhejiang'],help="place chosen to predict (can only be used when mode is predict)")

    
# 设置随机种子，以确保结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)





def create_sequences(data, seq_length):
    xs = []
    ys = []
 
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
 
    return np.array(xs), np.array(ys)


def loaddata(path):
    with open(path,"r") as f:
        #读取数据
        data=pd.read_csv(path)
        #data=np.array(data)
        total_d=len(data)
        
        # test_data_size = 889
        # train_data = data.iloc[:test_data_size]
        # test_data = data.iloc[test_data_size:]
        # 设置序列长度
        seq_length = 10
        
        
        # 为了提高模型的训练速度和性能，我们必须缩放数据（值将在 0 和 1 之间）。
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        data = scaler.transform(data)
        
        #按8:2比例划分训练集和测试集
        x_train,y_train,x_test,y_test=[],[],[],[]
        
        for i in range(total_d-seq_length-1):
            if np.random.random()<0.8:
                x_train.append(data[i:(i+seq_length)])
                y_train.append(data[i+seq_length])
            else:
                x_test.append(data[i:(i+seq_length)])
                y_test.append(data[i+seq_length])
                
        x_train,y_train=np.array(x_train),np.array(y_train)    
        x_test,y_test=np.array(x_test),np.array(y_test)
        #x_train,y_train=create_sequences(train_data,7)
        #x_test,y_test=create_sequences(test_data,7)
 
        
        # 将数据转换为PyTorch张量
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
        
    return x_train,y_train,x_test,y_test



# 把模型封装到一个自torch.nn.Module的类中    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.2)  
        self.fc = nn.Linear(hidden_size, output_size)  
  
    def forward(self, x):  
        # 初始化隐藏状态和单元状态  
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  
  
        # LSTM输出包含输出序列、隐藏状态和单元状态  
        out, _ = self.lstm(x, (h0, c0))  
  
        # 取最后一个时间步的隐藏状态作为全连接层的输入  
        out = self.fc(out[:, -1, :])  
        return out  



# 构建一个用于训练模型的辅助函数
def train_model(
        model,
        args,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        
):
    loss_fn = torch.nn.MSELoss(reduction="sum")
     
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = args.num_epochs
     
    #loss_fn=loss_fn.to(device)

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)
 
    print("Start training...")
    for t in tqdm(range(num_epochs)):
        #model.reset_hidden_state()
        y_pred = model(x_train)
        loss = loss_fn(y_pred.float(), y_train)
 
        if x_test is not None:
            with torch.no_grad():
                y_test_pred = model(x_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()
 
            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')
 
        train_hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
    print("Save checkpoint...")
    # 保存checkpoint  
    checkpoint = {  
        'model_state_dict': model.state_dict(),  
        'optimizer_state_dict': optimiser.state_dict(),  
        'epoch': num_epochs, 
        'loss':str(loss.item())
    }  
    # 使用torch.save保存checkpoint  
    torch.save(checkpoint, args.checkpoint_path)
    print("finished!")
    return model.eval(), train_hist, test_hist


#加载模型
def load_model(model,checkpoint_path):
    print("load checkpoint..")
    checkpoint=torch.load(checkpoint_path)
    print("MSE={}".format(checkpoint['loss']))
    model.load_state_dict(checkpoint['model_state_dict'])   
    return model.eval()


pic_length=10
seq_length1=10


#模型预测
def model_predict(model,whole_data,j):     #j：指定位置开始预测，pic_length:预测多长
    print("use model to predict..")
    scaler = MinMaxScaler()
    scaler=scaler.fit(whole_data)
    whole_data=scaler.transform(whole_data)
    y_true,y_preds=[],[]
    for i in range(j,j+pic_length):
        y_true.append(whole_data[seq_length1+i])
    y_true=scaler.inverse_transform(y_true).astype(int)
    #x_test = torch.from_numpy(np.array(x_test)).float()
    with torch.no_grad():
        for i in range(j,j+pic_length):
            x=[whole_data[i:seq_length1+i]]
            x = torch.from_numpy(np.array(x)).float()
            y_pred=model(x)
            y_pred=scaler.inverse_transform(y_pred)
            y_pred=y_pred.astype(int)
            y_preds.append(y_pred[0])
            
    return y_preds,y_true

#模型评估
def calMAE(y_preds,y_trues):
    print("test model..")
    with open("../../autodl-tmp/china_daily_confirmed.csv","r") as f:
         whole_data=pd.read_csv(f)
    scaler=MinMaxScaler()
    scaler=scaler.fit(whole_data)
    n=len(y_preds)
    mae=0
    for i in range(n):
        yp,yt=y_preds[i],y_trues[i]
        yp,yt=[yp.detach().numpy()],[yt.detach().numpy()]
        yp,yt=scaler.inverse_transform(yp).astype(int),scaler.inverse_transform(yt).astype(int)
        abe=np.abs(yt-yp)
        mae+=np.sum(abe)/33/n
    return mae
    
def drawPlot(y_true,y_pred,i,whole_data,title=''):#i:选择的省份的序列编号
    x=range(40,50)
    xy=range(50,60)
    wd_dict=list(whole_data.columns)
    y_pre=whole_data[i]
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    plt.plot(x,y_pre[list(x)],label="days_before",color="#FF0000",marker="o",linestyle='-')
    plt.plot(xy, y_true[:,wd_dict.index(i)], label="y_true",color="#00FF00", marker='o', linestyle="-")
    plt.plot(xy,y_pred[:,wd_dict.index(i)],label="y_predict",color="#0000FF",marker="*", linestyle="-")
    plt.xlabel('time series')
    plt.ylabel('value')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(title)
    
if __name__=='__main__':
    args=parser.parse_args()
    x_train,y_train,x_test,y_test=loaddata(args.input_path)
    LSTMmodel=LSTM(33,512,3,33)
    #训练模型
    if args.mode=="train":
        model,train_hist, test_hist = train_model(
                  LSTMmodel,
                  args,
                  x_train,
                  y_train,
                  x_test,
                  y_test
        )
    #评估模型
    elif args.mode=="test":
        load_model(LSTMmodel,args.checkpoint_path)
        y_pred=LSTMmodel(x_test)
        print(calMAE(y_pred,y_test))
        pass
    #使用模型进行预测
    else:
        load_model(LSTMmodel,args.checkpoint_path)
        #模型评估
        #y_pred=LSTMmodel(x_test)
        #print(calMAE(y_pred,y_test))
    
        #使用模型进行预测，预测后10天的效果
        with open("../../autodl-tmp/china_daily_confirmed.csv","r") as f:
            whole_data=pd.read_csv(f)
       
        pred_y,true_y=model_predict(LSTMmodel,whole_data,40)
        place=args.place
        drawPlot(true_y,pred_y,place.replace("_",' '),whole_data,place+'_confirmed_predict')
    