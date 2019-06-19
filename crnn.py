import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Bidirectional recurrent neural network
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        '''
        输入数据格式： 
        input(seq_len, batch, input_size) 
        h0(num_layers * num_directions, batch, hidden_size) 
        c0(num_layers * num_directions, batch, hidden_size)

        输出数据格式： 
        output(seq_len, batch, hidden_size * num_directions) 
        hn(num_layers * num_directions, batch, hidden_size) 
        cn(num_layers * num_directions, batch, hidden_size)
        '''
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(1), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(1), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        # out: tensor of shape (seq_len, batch_size, hidden_size*2)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x, (h0, c0))
        
        seq_len, batch_size, hidden_size = out.size()
        
        out = self.fc(out.view(seq_len * batch_size, hidden_size))
        
        # reshape for ctc
        return out.view(seq_len, batch_size, -1)

class CRNN(nn.Module):
    '''
    CRNN model
    '''
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.cnn = nn.DataParallel(self.cnn)
        
        self.rnn = nn.Sequential(
            BiRNN(input_size=512, hidden_size=256, num_layers=2, num_classes=num_classes))

    def forward(self, input):
        x = self.cnn(input) #[b, c, h=1, w]
        x = x.squeeze(2).permute(2, 0, 1) # [w, b, c]
        output = self.rnn(x)
        return output


if __name__ == '__main__':
    crnn = CRNN(5).to(device)
    print(crnn)