#%%
import math
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#%%
def f(x):
    return 3*x**2 - 4*x +5
# %%
f(3.0)
# %%
xs = np.arange(-5, 5, 0.25)
ys = f(xs)
plt.plot(xs, ys)
# %%
h = 0.001
x = 3.0
(f(x+h)-f(x))/h
# %%
a = 2
b = -3.0
c = 10.0
d = a*b+c
d
# %%
h = 0.0001
a = 2
b = -3.0
c = 10.0
d1 = a*b+c
a+=h
d2 = a*b+c
d1, d2, (d2-d1)/h
# %%
class Value:
    def __init__(self, data, _children=(), op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev=set(_children)
        self.op = op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        out= Value(self.data+other.data,(self,other),op="+")
        def _backward():
            self.grad=1.0*out.grad
            other.grad=1.0*out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        out= Value(self.data*other.data,(self,other),op="*")
        def _backward():
            self.grad=other.data*out.grad
            other.grad=self.data*out.grad
        out._backward = _backward
        return out
    def _rmul__(self, other):
        return self*other
    def __truediv__(self, other):
        return self*other**(-1)
    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self),op="tanh")
        return out
    def _backward():
        self.grad+=other * (self.data **(other-1))*out.grad
    out._backward = _backward
    return out
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out=Value(self.data**other,(self),op="**")
    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self),op="exp")
        return out
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
e=a*b; e.label="e"
d=e+c; d.label="d"
f = Value(-2.0, label="f")
L=d*f; L.label="L"
d
# %%
class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    
    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward
    
    return out

  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    
    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward
      
    return out
  
  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')
    
    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward
    
    return out
  
  def backward(self):
    
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
      node._backward()


a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'
d = e + c; d.label = 'd'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
L
# %%

# %%
def lol():
  
  h = 0.001
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L1 = L.data
  
  a = Value(2.0, label='a')
  b = Value(-3.0, label='b')
  b.data += h
  c = Value(10.0, label='c')
  e = a*b; e.label = 'e'
  d = e + c; d.label = 'd'
  f = Value(-2.0, label='f')
  L = d * f; L.label = 'L'
  L2 = L.data
  
  print((L2 - L1)/h)
  
lol()
# %%
plt.plot(np.arange(-5, 5, 0.25), np.tanh(np.arange(-5, 5, 0.25)))
plt.grid()
# %%
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(.7, label='b')
x1w1 = x1*w1; x1w1.label = 'x1w1'
x2w2 = x2*w2; x2w2.label = 'x2w2'
x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label = 'x1w1x2w2'
n = x1w1x2w2+b; n.label = 'n'
o=n.tanh(); o.label = 'o'