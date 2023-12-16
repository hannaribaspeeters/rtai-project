import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import time

from networks import get_network
from utils.loading import parse_spec

DEVICE = "cpu"

def debug(*args, **kwargs):
    # print(*args, **kwargs)
    pass

class RelationalConstraint:
    def __init__(self, weight, bias, lower=True): 
        super().__init__()
        self.weight = weight # (out_features, in_features)
        self.bias = bias # (out_features)
        self.lower = lower # Only used for printing

        assert self.weight.shape[0] == self.bias.shape[0]
        assert len(self.weight.shape) == 2 # (out_features, in_features)
        assert len(self.bias.shape) == 1 # (out_features)

    def __repr__(self) -> str:
        in_features = self.weight.shape[1]
        out_features = self.weight.shape[0]
        out = ""
        for j in range(out_features):
            out += f"a_{j}>=" if self.lower else f"a_{j}<="
            for i in range(in_features):
                if self.weight[j,i] != 0:
                    out += f"{self.weight[j,i]}*x_{i} + "
            out += f"{self.bias[j]}\n"
        return out

    def evaluate(self, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        Wmax = torch.max(torch.zeros_like(self.weight), self.weight)
        Wmin = torch.min(torch.zeros_like(self.weight), self.weight)
        if self.lower:
            return self.bias + Wmax @ lb + Wmin @ ub
        return self.bias + Wmax @ ub + Wmin @ lb
    
class Shape:
    def __init__(self, lb: torch.Tensor, ub: torch.Tensor, rel_lb: RelationalConstraint=None, rel_ub: RelationalConstraint=None, parent=None):
        self.lb = lb
        self.ub = ub
        assert (self.lb <= self.ub).all()
        self.rel_lb = rel_lb
        self.rel_ub = rel_ub
        if self.rel_ub is not None:
            self.rel_ub.lower = False
        self.parent = parent

        # Resolved is only used for testing script in a notebook. Will store the result. And make sure 
        # that we don't backsubstitute twice, in normal analysis this is not done.
        self._resolved = False

        self._input = False
        if parent is None:
            self._input = True
            self._resolved = True

    def __repr__(self) -> str:
        return f"Shape(lb={[round(el, 3) for el in self.lb.tolist()]},ub={[round(el, 3) for el in self.ub.tolist()]})" + (f"\n{self.rel_lb}" if self.rel_lb is not None else "") + (f"\n{self.rel_ub}" if self.rel_ub is not None else "")

    def backsubstitute(self):
        if self._resolved:
            return
        rel_lb = self.rel_lb
        rel_ub = self.rel_ub
        node = self.parent

        assert (self.lb <= self.ub).all()

        while(not node._input):
            # lower bound
            W, b = rel_lb.weight, rel_lb.bias
            Wmax = torch.max(torch.zeros_like(W), W)
            Wmin = torch.min(torch.zeros_like(W), W)

            rel_lb_W = Wmax @ node.rel_lb.weight + Wmin @ node.rel_ub.weight
            rel_lb_b = b + Wmax @ node.rel_lb.bias + Wmin @ node.rel_ub.bias
            rel_lb = RelationalConstraint(rel_lb_W, rel_lb_b)

            # upper bound
            rel = rel_ub
            W, b = rel.weight, rel.bias
            Wmax = torch.max(torch.zeros_like(W), W)
            Wmin = torch.min(torch.zeros_like(W), W)
            rel_ub_W = Wmax @ node.rel_ub.weight + Wmin @ node.rel_lb.weight
            rel_ub_b = b + Wmax @ node.rel_ub.bias + Wmin @ node.rel_lb.bias
            rel_ub = RelationalConstraint(rel_ub_W, rel_ub_b, lower=False)

            node = node.parent

        # resolve the relational constraints with respect to the input
        self.lb = torch.max(rel_lb.evaluate(node.lb, node.ub), self.lb)
        self.ub = torch.min(rel_ub.evaluate(node.lb, node.ub), self.ub)

        # check for validity
        assert self.lb.isfinite().all()
        assert self.ub.isfinite().all()
        assert (self.lb <= self.ub).all()
        self._resolved = True



class DeepPolyBase(torch.nn.Module):
    def __init__(self, verbose=False, name="", requires_backsubstitute=False):
        super().__init__()
        self.verbose = verbose
        self.name = name
        self.rel_lb = None
        self.rel_ub = None
        self.requires_backsubstitute = requires_backsubstitute

    def print(self, out):
        if self.verbose:
            print(self.name)
            print(out)
    
    def forward(self, x: Shape) -> Shape:
        return x

    def _setup(self, x: Shape) -> None:
        """
        Setup the relational constraints for the layer. This is called once before the first forward pass.
        """
        raise NotImplementedError()
    
    def setup(self, x: Shape) -> Shape:
        if self.requires_backsubstitute:
            x.backsubstitute()
            self._setup(x)
        if self.rel_lb is None or self.rel_ub is None:
            self._setup(x)
        return x        


class DeepPolyLinear(DeepPolyBase):
    def __init__(self, layer: nn.Linear, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.weight = layer.weight
        self.bias = layer.bias

    def _setup(self, x: Shape) -> None:
        self.rel_lb = RelationalConstraint(self.weight.data, self.bias.data)
        self.rel_ub = RelationalConstraint(self.weight.data, self.bias.data, lower=False)

    def forward(self, x: Shape) -> Shape:
        self.setup(x)

        c = (x.lb+x.ub)/2.
        r = (x.ub-x.lb)/2.
        c = torch.matmul(self.weight,c)+self.bias
        r = torch.matmul(torch.abs(self.weight),r)

        out = Shape(c-r, c+r, rel_lb=self.rel_lb, rel_ub=self.rel_ub, parent=x)
        self.print(out)
        return out
    
def outputShapeConv2d(input_len,kernel_shape,padding,stride):
    """
    Compute the output shape of a 2D convolution operation image.
    Parameters:
    input_len (int): The hight/width of the input image.
    kernel_shape (torch.Tensor): The convolution kernel of shape (out_channels, in_channels, kernel_height, kernel_width).
    padding (int,int): The padding size tuple
    stride (int,int): The stride size tuple
    Returns:
    (int,int): The output shape of the convolution operation.
    """
    h=w=input_len
    _,_,kh,kw = kernel_shape
    ph,pw = padding
    sh,sw = stride
    output_height = int((h + ph+pw - kh) / (sh) + 1)
    output_width= int((w + pw+ph - kw) / (sw) + 1)
    return (output_height,output_width)

def conv2dToAfinneBase(image_size,kernel,stride):
    """
    Compute the convolution as a matrix multiplication without padding.
    Parameters:
    image_size (int,int): The size of the input image (length_image,number_of_channels).
    kernel (torch.Tensor): The convolution kernel of shape (out_channels, in_channels, kernel_height, kernel_width).
    stride (int,int): The stride size tuple
    Returns:
    torch.Tensor: The convolution as a matrix multiplication without padding of shape (out_channels,in_channels,output_image_hight,length_image)
    """
    n,c = image_size
    sw,sh = stride
    assert(sw==sh) # Assume same stride
    h=w=int(np.sqrt(n))
    dim_out,_,kh,kw = kernel.shape #Assume square kernel
    assert(kh==kw)
    output_shape_h = outputShapeConv2d(h,kernel.shape,(0,0),stride)[0]

    #Add base row
    base_row = torch.zeros(dim_out,c,1,h*w)
    for i in range(kw):
        base_row[:,:,0,i*w:i*w+kw] = kernel[:,:,i,:]

    #Repeat base row with an offset to create first pass of the kernel (first output row)
    base_matrix = torch.zeros((dim_out,c,output_shape_h,h*w))
    for i in range(output_shape_h):
        base_matrix[:,:,i,:] = torch.roll(base_row,i*sw)[:,:,0,:]
    output = base_matrix

    #Repeat the base matrix with an offset to create the rest of the passes of the kernel (other output rows)
    for i in range(output_shape_h-1):
        output = torch.cat((output,torch.roll(base_matrix,sw*(i+1)*w)),dim=2)
    return output

#works only for square images and square kernels
def conv2dToAfinne(image_len,kernel,padding,stride):
    """
    Compute the convolution as a matrix multiplication.
    Parameters:
    image_len torch.tensor: The len of the input image, image is flattened.
    kernel (torch.Tensor): The convolution kernel of shape (out_channels, in_channels, kernel_height, kernel_width).
    padding (int,int): The padding size tuple
    stride (int,int): The stride size tuple
    Returns:
    torch.Tensor: The convolution as a matrix multiplication of shape (output_length,length_image)
    """
    dim_out,dim_in,kh,kw = kernel.shape #Assume square kernel
    assert(kh==kw)
    n = image_len[-1]
    h=w = int(np.sqrt(n/dim_in))

    padded_image = transforms.Pad(padding)(torch.zeros(dim_in,h,w))
    #Create fake image to get the non padded index
    non_padded_index = (transforms.Pad(padding)(torch.ones((h,w))).view(-1) !=0).nonzero().view(-1)
    #Do the convolution with padded image
    stacked_kernels = conv2dToAfinneBase(padded_image.view(-1,dim_in).shape,kernel,stride)
    #Remove the padded columns
    stacked_kernels = stacked_kernels[:,:,:,non_padded_index]
    #Reshape to get the correct width, adding 0 columns to match output size
    missing_columns = n-stacked_kernels.shape[-1]
    stacked_kernels = torch.nn.functional.pad(stacked_kernels, (0,missing_columns), value=0)
    #Create offset to align kernels with their corresponding channels
    for i in range(stacked_kernels.shape[1]):
        length_image = int(h*w)
        stacked_kernels[:,i,:,:] = torch.roll(stacked_kernels[:,i,:,:],i*length_image)
    stacked_kernels = stacked_kernels.view(dim_out,-1,stacked_kernels.shape[2],n).sum(dim=1)
    return stacked_kernels.reshape(-1,n)

class DeepPolyConv2d(DeepPolyBase):
    def __init__(self, layer: torch.nn.Conv2d, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Reshape the weight and bias to be compatible with the linear layer
        self.layer = layer

    def _setup(self, x: Shape) -> None:
        self.weight = conv2dToAfinne(x.lb.shape,self.layer.weight,self.layer.padding,self.layer.stride)
        dim_in = self.layer.weight.shape[1]
        h = w = int(np.sqrt(x.lb.shape[-1]/dim_in))
        output_h,output_w = outputShapeConv2d(h,self.layer.weight.shape,self.layer.padding,self.layer.stride)
        self.bias = self.layer.bias.unsqueeze(1).repeat(1,output_w*output_h).flatten()

        self.rel_lb = RelationalConstraint(self.weight.data, self.bias.data)
        self.rel_ub = RelationalConstraint(self.weight.data, self.bias.data, lower=False)


    def forward(self, x: Shape) -> Shape:
        self.setup(x)

        c = (x.lb+x.ub)/2.
        r = (x.ub-x.lb)/2.

        c = self.weight @ c + self.bias
        r = torch.abs(self.weight) @ r

        out = Shape(c-r,c+r, rel_lb=self.rel_lb, rel_ub=self.rel_ub, parent=x)
        self.print(out)
        return out


class DeepPolyLeakyReLU(DeepPolyBase):
    def __init__(self, layer: torch.nn.LeakyReLU, initialization="alpha", *args, **kwargs):
        super().__init__(*args, requires_backsubstitute=True, **kwargs)
        self.layer = layer
        self.initialization = initialization
        self.beta = None
        self.negative_slope = layer.negative_slope

    def init_beta(self, x: Shape) -> None:
        alfa = self.negative_slope
        # in the case alfa<1 beta is in [alfa, 1]
        # in the case alfa>1 beta is in [1, alfa]
        # I initialize beta to alfa and from there we have to see how we optimize
        
        beta = torch.full(x.lb.size(), alfa)
        beta = torch.where(x.ub <= -x.lb, beta, torch.ones_like(x.lb))
        # random initialization between alfa and 1
        #beta = torch.rand(x.lb.size())*(1-alfa)+alfa

        beta = beta.float()
        beta = torch.nn.Parameter(beta)
        beta.requires_grad = True
        self.beta = beta

    def _setup(self, x: Shape) -> None:
        alfa = self.layer.negative_slope
        x.backsubstitute()
        # Case 1: Below-zero -> propagate(linear with slope=alfa)
        case1 = (x.ub < 0.)
        W_l, b_l = torch.full(x.lb.size(), alfa), torch.zeros_like(x.lb)
        W_u, b_u = torch.full(x.lb.size(), alfa), torch.zeros_like(x.lb)

        # Case 2: Above-zero -> propagate
        case2 = (x.lb > 0.)
        W_l, b_l = torch.where(case2, torch.ones_like(x.lb), W_l), b_l
        W_u, b_u = torch.where(case2, torch.ones_like(x.lb), W_u), b_u

        # Case 3: crossing
        crossing = ~case1 & ~case2      
        if self.beta is None:
            self.init_beta(x)
        else:
            # clamp beta to alfa
            if alfa < 1:
                self.beta.data.clamp_(min=alfa, max=1)
            else:
                self.beta.data.clamp_(min=1, max=alfa)     
    
        W_l = torch.where(crossing, self.beta, W_l)
        b_l = torch.where(crossing, torch.zeros_like(x.lb), b_l)

        W_u = torch.where(crossing, torch.div(x.ub-alfa*x.lb, x.ub-x.lb), W_u)
        b_u = torch.where(crossing, torch.div(x.ub*x.lb*(alfa-1), x.ub-x.lb), b_u)

        self.rel_lb = RelationalConstraint(torch.diag(W_l), b_l)
        self.rel_ub = RelationalConstraint(torch.diag(W_u), b_u, lower=False)


    def forward(self, x: Shape) -> Shape:
        self.setup(x)
        if self.layer.negative_slope <= 1:
            out = Shape(F.leaky_relu(x.lb, self.layer.negative_slope), F.leaky_relu(x.ub, self.layer.negative_slope), rel_lb=self.rel_lb, rel_ub=self.rel_ub, parent=x)
        else:
            out = Shape(F.leaky_relu(x.lb, self.layer.negative_slope), F.leaky_relu(x.ub, self.layer.negative_slope), rel_lb=self.rel_ub, rel_ub=self.rel_lb, parent=x)
        self.print(out)

        return out

class DeepPolyReLu(DeepPolyLeakyReLU):
     def __init__(self, layer: torch.nn.ReLU, *args, **kwargs):
         setattr(layer, "negative_slope", 0)
         super().__init__(layer, *args, **kwargs)


class VerificationHead(DeepPolyLinear):
    """
    Linear layer that verifies that the true class is the largest output. Constraint is verified if
    all values of the output shape lower bound are positive.

    Basically calculates
    x_true - x_other for all classes (and just x_true for the true class)
    """
    def __init__(self, num_classes: int, true_class=False) -> None:
        super().__init__(layer=nn.Linear(num_classes,num_classes))
        self.true_class = true_class
        weight = -torch.eye(num_classes)
        weight[:,true_class] +=1
        self.weight.data = weight
        self.bias.data = torch.zeros(num_classes)
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x: Shape) -> Shape:
        return super().forward(x)
    
class VerifierLoss(nn.Module):
    """
    Loss function for the verification head. Returns 0 if the constraint is verified, otherwise
    returns the maximum violation.
    """
    def __init__(self, metric: str) -> None:
        super().__init__()
        self.metric = metric
    
    def forward(self, x: Shape) -> torch.Tensor:
        if self.metric == "l2":
            return torch.norm(x.ub - x.lb, p=2)
        violations = torch.abs(x.lb[x.lb <= 0])
        if len(violations) == 0:
            return torch.zeros(1)
        if self.metric == "max": # maximum violation    
            return torch.max(violations)
        if self.metric == "sum": # sum of violations
            return torch.sum(violations)
        raise ValueError(f"Unknown metric {self.metric}")
        
    

def create_analyzer(net: nn.Module, verbose=False):
    layers = []
    for name, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            layers.append(DeepPolyLinear(layer, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.ReLU):
            layers.append(DeepPolyReLu(layer, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.Conv2d):
            layers.append(DeepPolyConv2d(layer, name=name, verbose=verbose))
        if isinstance(layer, torch.nn.LeakyReLU):
            layers.append(DeepPolyLeakyReLU(layer, name=name, verbose=verbose))
    return nn.Sequential(*layers)


def analyze(
    net: torch.nn.Module, inputs: torch.Tensor, eps: float, true_label: int, min: float = 0, max: float = 1, use_time_limit=False, max_epochs=50) -> bool:
    start = None
    if use_time_limit:
        start = time.time()
        timeout = 20 # seconds
    # Create validation head
    num_classes = net[-1].out_features
    verification_head = VerificationHead(num_classes, true_label)

    # turn off gradients
    for param in net.parameters():
        param.requires_grad = False

    # create analyser net and add verification_head
    analyzer_net = nn.Sequential(*create_analyzer(net, verbose=False), verification_head)

    lb = inputs - eps
    ub = inputs + eps
    lb.clamp_(min=min, max=max)
    ub.clamp_(min=min, max=max)

    # Flatten input
    lb = lb.view(-1)
    ub = ub.view(-1)
    input_shape = Shape(lb, ub)

    loss = VerifierLoss("sum")

    # Beta training loop
    epoch = 0
    analyzer_net.train()

    # First push to initialize the relational constraints and parameters
    output_shape = analyzer_net(input_shape)
    output_shape.backsubstitute()

    result = (output_shape.lb > 0).all().item()
    if result:
        return result
    
    # count number of parameters
    num_params = 0
    for param in analyzer_net.parameters():
        num_params += param.numel() if param.requires_grad else 0

    # if we have no params, we can just test the output
    if num_params == 0:
        debug("No parameters to train.")
        output_shape = analyzer_net(input_shape)
        output_shape.backsubstitute()

        result = (output_shape.lb >= 0).all().item()
        if result:
            return result
        return False
    
    debug(f"Number of parameters: {num_params}")
    # Train beta
    optimizer = torch.optim.Adam(analyzer_net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, 1, eta_min=0.0001)
    while (start is None or time.time() - start < timeout):
        optimizer.zero_grad()
        output_shape = analyzer_net(input_shape)
        output_shape.backsubstitute()

        result = (output_shape.lb >= 0).all().item()
        if result:
            debug(f"Epoch {epoch}: Loss:", loss(output_shape).item())
            return result
        
        loss_value = loss(output_shape)
        loss_value.backward()
        optimizer.step()
        scheduler.step()
        debug(f"Epoch {epoch}: Loss: {loss_value.item()}: LR: {scheduler.get_last_lr()[0]}", end="\r")
        epoch += 1
        if use_time_limit and (epoch >= max_epochs and not max_epochs == -1):
            debug()
            break
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)
    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
