from DFN import *
from FEN import *

class SAFNet(nn.Module):
    def __init__(self):
        super(SAFNet, self).__init__()

        self.DFN = DFN()
        self.FEN = FEN()

    def forward(self, x, y):
        output_DFN = self.DFN(x, y)
        output_FEN = self.FEN(output_DFN)

        return output_FEN