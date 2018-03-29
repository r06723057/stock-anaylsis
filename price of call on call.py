from math import *
from scipy.stats import norm, multivariate_normal
import numpy as np
from scipy.optimize import newton
s0=100.0   #s0 is stock price at time 0
r=0.05     #
g=0.2      #g is sigma
t1=1.0
t2=1.5
k2=6.0     #exercise price on call
q=0.03
k1=100.0   #exercise price on call on call

def f(x):
    t3=t2-t1
    d1=(log(x/k2)+(r-q+(g**2.0)/2.0)*t3)/g/(t3**0.5)
    d2=d1-g*(t3**0.5)
    n1=norm(0, 1).cdf(d1)
    n2=norm(0, 1).cdf(d2)
    return (x*n1*e**(-q*t3)-(k2*n2*e**(-r*t3))-k1)


ss=newton(f,k1,tol=1e-16, maxiter=1000)  #ss is the stock price that make the call price(exercise price of that is k2 at time t2) is k1 at time t1


a1=(log(s0/ss)+(r-q+g**2/2)*t1)/g/(t1**0.5)
a2=a1-g*(t1**0.5)
b1=(log(s0/k2)+(r-q+g**2/2)*t2)/g/(t2**0.5)
b2=b1-g*(t2**0.5)

mean = [0,0]
cov= [[1, (t1/t2)**0.5 ],[(t1/t2)**0.5, 1]]
m = multivariate_normal(mean=mean, cov=cov)
M1=m.cdf(np.array([a1,b1]))
M2=m.cdf(np.array([a2,b2]))
n22=norm(0, 1).cdf(a2)



c=s0*(e**(-q*t2))*M1-k2*(e**(-r*t2))*M2-(e**(-r*t1))*k1*n22 #c is call on call price
print(c)
