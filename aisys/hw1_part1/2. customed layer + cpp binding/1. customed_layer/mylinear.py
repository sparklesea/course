import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

class myLinearFunction(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())#mm表示矩阵乘法
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)#t表示转置
        return grad_input, grad_weight

                  
class myLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(myLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return myLinearFunction.apply(input, self.weight)