"""Horner-polynomial neural-network models for ODE demonstrations.

The models in this file represent a scalar function y(t) with a polynomial
evaluated by Horner's scheme. Some variants also embed initial conditions
directly into the polynomial construction. Those are "hard-IC" models: the
initial condition is satisfied by construction, so the training loss can focus
on the ODE residual.
"""

import torch
import torch.nn as nn
import math


class myBias1(nn.Module):
    """Learnable additive coefficient used inside the Horner recurrence.

    The layer computes

        output = input + b * out_scale

    where `b` is trainable. The parameter is initialized from a uniform
    distribution centered at `rand_dist_mean` with half-width `rand_dist_range`.
    With the default arguments, coefficients start close to zero.
    """

    def __init__(self, in_features: int, out_features: int, out_scale = 1., 
                 rand_dist_range = 0.01, rand_dist_mean = 0):
        super(myBias1, self).__init__()
        self.b = nn.Parameter(2*rand_dist_range*torch.rand(out_features,in_features, dtype = torch.float32).view(out_features,in_features) - rand_dist_range + rand_dist_mean, requires_grad = True)        
        self.out_scale = out_scale     
        
    def forward(self, x):
        """Add the scaled trainable coefficient to `x`."""

        return x + (self.b)*self.out_scale


class myLinear1(nn.Module):
    """Learnable multiplicative coefficient used as the leading Horner term.

    The layer computes

        output = matmul(input, a) * out_scale

    where `a` is trainable. For these one-dimensional examples, this is a
    scalar multiplication, but the shape arguments keep the layer general.
    """

    def __init__(self, in_features: int, out_features: int, out_scale = 1., 
                 rand_dist_range = 0.01, rand_dist_mean = 0):
        super(myLinear1, self).__init__()
        self.a = nn.Parameter(2*rand_dist_range*torch.rand(out_features,in_features, dtype = torch.float32).view(out_features,in_features) - rand_dist_range + rand_dist_mean, requires_grad = True)        
        self.out_scale = out_scale
               
    def forward(self, x):
        """Apply the scaled trainable multiplication to `x`."""

        return torch.matmul(x,self.a)*self.out_scale


class Horner(nn.Module):
    """Plain Horner polynomial model without embedded initial conditions.

    Arguments:
        order: Polynomial order.
        a, b: Input interval. The input is scaled from [a, b] to [-1, 1].

    Conceptually, this represents a polynomial evaluated in nested Horner form.
    This class is useful as a base polynomial model, while the `Horner_IC_*`
    classes below add hard initial-condition constraints.
    """

    def __init__(self, order, a, b):
        super(Horner, self).__init__()
        
        self.a = a
        self.b = b
        self.order = order
        
        # Leading coefficient a[n].
        self.linear = myLinear1(1, 1)#, 1/math.factorial(order)) # A linear module for a[n]

        # Remaining coefficients a[n-1], ..., a[0].
        self.biases = nn.ModuleList() # Init. of list containing all biases a[n-1], ..., a[0]
        
        for i in range(order):
            self.biases.append( myBias1(1, 1))#  ,1/math.factorial(order-1-i)) )
        
      
    def forward(self, x): 
        """Evaluate the polynomial at input coordinates `x`."""

        # Scale the physical ODE domain [a, b] to the polynomial domain [-1, 1].
        x = 2/(self.b-self.a)*x - (self.b+self.a)/(self.b-self.a) # Scaling from [a,b] to [-1,1]
       
        # Horner recurrence:
        # a[n]*x -> a[n-1] + a[n]*x -> a[n-2] + x*(...) -> ...
        x1 = self.linear(x) # a[n]*x
        x1 = self.biases[0](x1) # a[n-1] + a[n]*x
        for i in range(1,self.order):
            x1 = self.biases[i](x*x1) # a[n-2] + x*(a[n-1] + a[n]*x) ... and so on
        
        return x1  
    

class Horner_IC_1_order(nn.Module):
    """Horner polynomial with a hard value initial condition.

    The physical interval [a, b] is scaled to [-1, 1], so the left endpoint
    t = a corresponds to x = -1. The forward pass adjusts the constant term so
    that the final polynomial satisfies

        y(a) = y0

    exactly, up to floating-point arithmetic. The loss in the first-order
    scripts therefore only needs the ODE residual.
    """

    def __init__(self, order, a, b):
        super(Horner_IC_1_order, self).__init__()
        
        self.a = a
        self.b = b
        self.order = order

        self.linear = myLinear1(1, 1) #
        self.biases = nn.ModuleList()
        
        for i in range(order-1):
            self.biases.append(myBias1(1, 1))
                               
    
    def forward(self, x, y0): # f(-1) = x0
        """Evaluate the hard-IC polynomial with left-endpoint value `y0`."""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = 2/(self.b-self.a)*x - (self.b+self.a)/(self.b-self.a) # Scaling from [a,b] to [-1,1]
        

        # Build the trainable nested polynomial part.
        x1 = self.linear(x) # a[n]*x
        x1 = self.biases[0](x1) # a[n-1] + a[n]*x
        for i in range(1,self.order-1):
            x1 = self.biases[i](x*x1) # a[n-2] + x*(a[n-1] + a[n]*x) ... and so on
            
        # Compute the correction needed at x=-1 so the polynomial value is y0.
        # This is the hard initial-condition embedding.
        xt = 0
        sgn  = 1
        for i in range(self.order-1-1,-1,-1):
            xt += sgn*self.biases[i](torch.zeros(1,1, dtype = torch.float32).to(device))
            sgn = sgn*(-1)
        xt += sgn*self.linear(torch.ones(1,1, dtype = torch.float32).to(device)) 

        
        x1 = x*x1 + y0 + xt
              
        return x1  
    
    
    
class Horner_IC_2_order(nn.Module):
    """Horner polynomial with hard value and derivative initial conditions.

    The physical interval [a, b] is scaled to [-1, 1]. The forward pass adjusts
    the two lowest-order coefficients so that the final polynomial satisfies

        y(a)  = y0
        y'(a) = y0d

    exactly, up to floating-point arithmetic. The derivative condition accounts
    for the scaling from [a, b] to [-1, 1].
    """

    def __init__(self, order, a, b):
        super(Horner_IC_2_order, self).__init__()
        
        self.a = a
        self.b = b
        self.order = order

        self.linear = myLinear1(1, 1) #
        self.biases = nn.ModuleList()    
        
        for i in range(order-2):
            self.biases.append(myBias1(1, 1))
                               
    
    def forward(self, x, y0, y0d): # f(-1) = x0
        """Evaluate the hard-IC polynomial with value `y0` and derivative `y0d`."""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = 2/(self.b-self.a)*x - (self.b+self.a)/(self.b-self.a) # Scaling from [a,b] to [-1,1]
        
        # Build the trainable nested polynomial part without the two IC terms.
        x1 = self.linear(x) # a[n]*x
        x1 = self.biases[0](x1) # a[n-1] + a[n]*x
        for i in range(1,self.order-2):
            x1 = self.biases[i](x*x1) # a[n-2] + x*(a[n-1] + a[n]*x) ... and so on
            

        # Compute the corrections for value and derivative at x=-1.
        # `xt` corrects the value and `xtt` corrects the first derivative.
        xt  = 0
        xtt = 0
        sgnd  = 1
        sgn = -1
        factord = 2
        for i in range(self.order-2-1,-1,-1):
            xtt += factord*sgnd*self.biases[i](torch.zeros(1,1, dtype = torch.float32).to(device))
            xt  += sgn*self.biases[i](torch.zeros(1,1, dtype = torch.float32).to(device))
            sgnd = sgnd*(-1)
            sgn = sgn*(-1)
            factord = factord + 1

        xtt += sgnd*factord*self.linear(torch.ones(1,1, dtype = torch.float32).to(device)) 
        xt += sgn*self.linear(torch.ones(1,1, dtype = torch.float32).to(device)) 

        # Convert the physical derivative y0d to the scaled x-domain and then
        # rebuild the two lowest-order coefficients.
        a1 = y0d*(self.b - self.a)/2  + xtt
        a0 = y0 + a1 + xt
        x1 = x1*x + a1
        x1 = x1*x + a0
              
        return x1      




        
