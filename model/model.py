
import torch  # 导入PyTorch库，用于构建和训练深度学习模型
import os  # 导入os库，用于与操作系统交互
import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理和分析
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from sklearn.preprocessing import MinMaxScaler  # 导入MinMaxScaler，用于数据标准化
from pandas.plotting import register_matplotlib_converters  # 导入register_matplotlib_converters，用于处理时间序列数据
from torch import nn, optim  # 导入神经网络和优化器模块

 
# 设置随机种子，以确保结果可复现
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 转换为PyTorch张量并移动到适当的设备上（例如，GPU）  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

seq_length1=10


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
        seq_length = seq_length1
        
        
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
class CoronaVirusPredictor(nn.Module):
 
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()
 
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
 
        self.lstm = nn.LSTM( 
            input_size=n_features,  # 输入特征维数：特征向量的长度，如 889
            # hidden_size 只是指定从LSTM输出的向量的维度，并不是最后的维度，因为LSTM层之后可能还会接其他层，如全连接层（FC），因此hidden_size对应的维度也就是FC层的输入维度。
            hidden_size=n_hidden,  # 隐层状态的维数：每个 LSTM 单元或者时间步的输出的 h(t) 的维度，单元内部有权重与偏差计算
            # num_layers 为隐藏层的层数，官方的例程里面建议一般设置为1或者2。
            num_layers=n_layers,  # RNN 层的个数：在竖直方向堆叠的多个相同个数单元的层数
            dropout=0.5  # 是否在除最后一个 LSTM 层外的 LSTM 层后面加 dropout 层
        )
 
        self.linear = nn.Linear(in_features=n_hidden, out_features=33)
 
    # 重置隐藏状态: 使用无状态 LSTM，需要在每个示例之后重置状态。
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )
 
    # 前向传播: 获取序列，一次将所有序列通过 LSTM 层。采用最后一个时间步的输出并将其传递给我们的线性层以获得预测。
    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
    
    
    
    
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
        checkpoint_path,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        
):
    loss_fn = torch.nn.MSELoss(reduction="sum")
     
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 300
     
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
    torch.save(checkpoint, checkpoint_path)
    print("finished!")
    return model.eval(), train_hist, test_hist


#加载模型
def load_model(model,checkpoint_path):
    print("load checkpoint..")
    checkpoint=torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])   
    return model.eval()


pic_length=800


#模型预测
def model_predict(model,whole_data,j):
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
        
    
def drawPlot(y_true,y_pred,i):
    x=range(pic_length)
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    plt.plot(x, y_true[:,i], label="true",color="#FF3B1D", marker='*', linestyle="-")
    plt.plot(x,y_pred[:,i],label="predict",color="#3399FF", marker='o', linestyle="-")
    plt.show()
    plt.savefig("plot.png")
    
if __name__=='__main__':
    #x_train,y_train,x_test,y_test=loaddata("../../autodl-tmp/china_daily_confirmed.csv")
    # model = CoronaVirusPredictor(
    #   n_features=33,
    #   n_hidden=512,
    #   seq_len=7,
    #   n_layers=3
    # )
    LSTMmodel=LSTM(33,512,3,33)
    #model=model.to(device)
#     model,train_hist, test_hist = train_model(
#       LSTMmodel,
#       "checkpoint/model_checkpoint.pth",
#       x_train,
#       y_train,
#       x_test,
#       y_test
#     )
    
    load_model(LSTMmodel,"checkpoint/model_checkpoint.pth")
    
    with open("../../autodl-tmp/china_daily_confirmed.csv","r") as f:
        whole_data=pd.read_csv(f)
       
        
    pred_y,true_y=model_predict(LSTMmodel,whole_data,0)
    drawPlot(true_y,pred_y,12)
    