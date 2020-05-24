from torch import nn
import torchvision

class ConvNet(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self.convLayer1 = nn.Conv3d(1,16,kernel_size=3, padding=1)
        self.activation1 = nn.Tanh()
        self.convLayer2 = nn.Conv3d(16,1,kernel_size=3, padding=1)


    def forward(self, x):
        # fill this in with calls to the defined conv. Layers
        # return the resulting transformed x

        out = self.convLayer1(x)
        out = self.activation1(out)
        out = self.convLayer2(out)
        return out

def Main():
    pass

if __name__ == '__main__':
    Main()
