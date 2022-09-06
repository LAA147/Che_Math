#!/usr/bin/env python
# coding: utf-8

# # Homework 1 - Basic practice in Jupyter and GitHub.
# 
# Programming can only be learned by doing. The following exercises blend coding math and text to create clearly defined results. For example, instead of calculating a simple number, you must print the number plus the units (e.g. 24 mg/L). 

# # Academic honesty statement
# 
# Name - Lakshmi Yasodhara Ananthabhotla
# 
# -   I worked alone on this assignment. I used internet and a few other resources to verify/learn the syntax.
# 

# # Problem 1: Gas Laws
# 
# 

# The virial equation for a gas can be represented as $\frac{PV}{RT} = 1 + \frac{B}{V} + \frac{C}{V^2}$ where $V$ is the molar volume.
# 
# For isopropanol $B=-388 cm^3/mol$ and $C=-26,000 cm^6/mol^2$.
# 
# We are going to find $V$ for isopropanol at 200 ∘C and 10 bar with a graphical technique.
# 
# In this problem $R=83.14 cm^3 bar / (mol K)$.
# 
# 

# ### Compute V for an ideal gas
# 
# 

# The ideal gas law is $1 = \frac{PV}{RT}$. Use this to estimate the volume of isopropanol in the ideal gas state. Print your answer with one decimal place and the printed answer **must include units.**
# 
# 

# In[13]:


R = 83.13 #cm3bar/(molK)
T = 200.0+273.0 #K
P = 10.0 #bar
V = (R*T)/P #cm3
print(round(V,1),"cm^3/mol")


# ### Compute V for the Virial Gas Law
# 
# 

# To do this, create a new function:
# 
# $f(V) = \frac{PV}{RT} - 1 - \frac{B}{V} - \frac{C}{V^2} = 0$
# 
# and then find values of $V$ where $f(V) = 0$. Start by defining this function and test that it works.Show that your function works by evaluating it for some examples, including an array of volumes.
# 
# 

# In[61]:


#import numpy as np
#import math
#from sympy.solvers import solve
from sympy import Symbol


def f(V):
   
    B = -388.0
    C = -26000.0 
    eq = ((P*V)/(R*T))-1-((B/V))-(C/(V**2))
    return eq

V1 = np.linspace(2000,5000,5) #Different values for volume in cm^3/mol

print(V1)

print(f(V1))
    


# ### Plot f(V) over a range where you can observe a zero
# 
# 

# You should make the x-axis sufficiently zoomed in to estimate the solution to about 10 cm<sup>3</sup>.
# 
# 

# In[64]:


import matplotlib.pyplot as plt
V2 = np.linspace(250,3900,1000)
plt.xlabel("Volume cm^3/mol")
plt.ylabel("f(V)")
plt.plot(V2,f(V2))
plt.plot(V2,V2*0)
#print(V)


# In[68]:


V2 = np.linspace(480,550,1000)
plt.xlabel("Volume cm^3/mol")
plt.ylabel("f(V)")
plt.plot(V2,f(V2))
plt.plot(V2,V2*0)
#print(V)


# In[71]:


V2 = np.linspace(3450,3510,1000)
plt.xlabel("Volume cm^3/mol")
plt.ylabel("f(V)")
plt.plot(V2,f(V2))
plt.plot(V2,V2*0)
#print(V)


# State in words where the solution(s) are. 
# 
# Solution 1 is in between 500 - 510 cm^3/mol 
# Solution 2 is in between 3480 - 3490 cm^3/mol
# 
# 

# ### Express this in the form of a cubic polynomial in $V$
# 
# 

# Derive an alternative expression for f(V) where it is a cubic polynomial of the form $0 = a V^3 + b V^2 + c V + d$. Write this expression in LaTeX, with explicit definitions for the coefficients.
# 
# 

# $\frac{PV}{RT} = 1 + \frac{B}{V} + \frac{C}{V^2}$
#  
# $\frac{PV}{RT} = \frac{V^2 + BV + C}{V^2}$
# 
# Cross-multiplying gives,
# 
# $PV^3 = RTV^2 + RTBV + CRT$
# 
# The final equation is,
# 
# $\frac{PV^3}{RT}-V^2-BV-C = 0$
# 
# The coefficients can be defined as,
# 
# $aV^3 + bV^2 + cV + d = 0$
# 
# where, $a = \frac{P}{RT}$, $b = -1$, $c = -B$ & $d= -C$

# # Problem 2: Running and plotting an ODE
# 
# Most programming is repeative. Once you learn how to run an ODE solver, running it again in the future involves just editing prior code. We discussed the Lorenz equations in class, now we'll run them ourselves. The equations are as follows: 
# 
# \begin{align}
# \dot{x} & = \sigma(y-x) \\
# \dot{y} & = \rho x - y - xz \\
# \dot{z} & = -\beta z + xy
# \end{align}
# 
# These equations are to describe a 2 dimensional layer of fluid that is heat from below and cooled from above. The derivation of these equations is beyond our interest here but can be found easily online. We want to focus on how to simulate such equations.
# 
# In the above, x is proportional to the intensity of the convective motion, while y is proportional to the temperature difference between the ascending and descending currents, similar signs of x and y denoting that warm fluid is rising and cold fluid is descending. The variable z is proportional to the distortion of vertical temperature profile from linearity, a positive value indicating that the strongest gradients occur near the boundaries.
# 
# Let the parameters be
# \begin{align}
# \sigma =10\\
# \beta =8/3\\
# \rho =28\\
# \end{align}
# 
# We will ignore units for this problem.
# 
# In the space below, 
# 1. Simulate this system with initial conditions of x = y = z = 1. Simulate out to time 1000 and take 0.1 time unit step sizes. Save the necessary results in a matrix.
# 

# In[77]:


import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

sig = 10.0
beta = 8.0 / 3.0
rho = 28.0 

def lorenz(state, t, sig, beta, rho): #defining function to return each derivative form
    x, y, z = state
    dxdt = sig * (y - x)
    dydt = (x * (rho - z)) - y
    dzdt = (x * y) - (beta * z)
     
    return [dxdt, dydt, dzdt]
 
x0 = 1.0
y0 = 1.0
z0 = 1.0

zero = [x0, y0, z0] #Initial condition array

l = (sig, beta, rho) #System parameters

t = np.linspace(0.0, 1000.0, 10000) #Defining time range and time steps


sol = odeint(lorenz, zero, t, args=(l)) #calculating the solution for the ODEs using the initial condition

p,q,r = sol.T 

print(sol.T)


# 2. Simulate the system again but with x = y = 1 and z = 0.9999 (precisely). Save the necessary results in a matrix.

# In[78]:



zero1 = [1.0, 1.0, 0.9999]

sol1 = odeint(lorenz, zero1, t, args=(l)) #Solving for a different initial condition

p1,q1,r1 = sol1.T

print(sol1.T)




# 3. On a single plot, plot x vs y from 1 and 2 above. Be sure to label axes.

# In[90]:



plt.xlabel("x")
plt.ylabel("y")
plt.plot(p,q, alpha = 0.8)



plt.plot(p1,q1, alpha = 0.7)

plt.legend(tuple(['Initial condition1 is x=y=z=1', 'Initial condition2 is x=y=1, z=0.9999']))


# 4. On a single plot, plot x vs z from 1 and 2 above. Be sure to label axes.

# In[91]:


plt.xlabel("x")
plt.ylabel("z")
plt.plot(p,r, alpha = 0.8)


plt.plot(p1,r1, alpha = 0.7)

plt.legend(tuple(['Initial condition1 is x=y=z=1', 'Initial condition2 is x=y=1, z=0.9999']))


# 5. On a single plot, plot x vs time from 1 and 2 above. Be sure to label.

# In[94]:


t = np.linspace(0,1000,10000)
plt.xlabel("t")
plt.ylabel("x")
plt.plot(t,p, alpha = 0.7)

plt.plot(t,p1, alpha = 0.35)

plt.legend(tuple(['Initial condition1 is x=y=z=1', 'Initial condition2 is x=y=1, z=0.9999']))


# 6. After completing all simulations, comment on how changing the initial value of z by 0.01% impacted the simulation outcomes. Be sure to use a Markdown cell for this.
# 
# Though there is a change in the output calcuated values of z, it cannot be visulalized to a great extent in the graph. Both the curves with the two different initial conditions seem to almost intersect.

# # Problem 3 - Reading COVID data and Prediction
# 
# I hope I mentioned in class that when it comes to addressing engineering questions computationally, you will often be expected to learn on your own. This means using google or what have you and searching for specific libraries that can help solve the problem at hand. 
# 
# Here, I have provided some data that I pulled from the New York Time's GitHub COVID-19 data repository. The file is called florida.csv and contains COVID outbreak data specific to Florida. The data contains: the week since the start of the outbreak, the 7 day sliding average of the number of COVID positive cases and the 7 day sliding average of the number of deaths reported.
# 
# In the cell below, write the code necessary to load this data into the workspace and give the resulting array the name "dat". The numpy.genfromtxt function may be helpful.

# In[87]:


pip install plotnine


# In[96]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd


data = pd.read_csv("florida.csv") #Reading the .csv file
data #Printing the data


# Next, make a points plot with the average number of cases on the x axis and the average number of deaths on the y axis. Have the color of the points change according to the week of the infection. There are a lot of ways to generate such a plot. Consider matlibplot and ggplot2. After plotting, create a markdown cell and comment on the how these two features have evolved in time.

# In[102]:


from plotnine import ggplot, aes, geom_point, geom_line
ggplot(data) + aes(x="cases_avg", y = "deaths_avg", color='week') + geom_point() #Scatter plot using ggplot method


# The ratio of average number of deaths to the average number of cases varied differently over the period. It was initially high for about 200 weeks and then went down for the next 100 weeks. Towards the end, the death rate went down though there were many cases being reported. This explains that there were peaks/waves during the pandemic. 

# One goal of every scientist should be to use data for predictions. Here, we'll use this goal as a chance to practice for loops. 
# 
# Ideally, in a world without delays and nonlinearities, the best way to predict an outcome is with a linear model, aka a straight line. Right a "for loop" which adds 4 straight lines to our plot of average infection and average mortality. Each line should have an intercept at zero and a slope of 0.001, 0.01, 0.1 and 1. 

# In[103]:


slope = [0.001, 0.01, 0.1, 1]

for i in slope:
    data[f"a{i}"]=i*data["cases_avg"]  #Creating columns in the data for the given slopes
    
data




# In[104]:


str(data)
ggplot(data) + geom_point(aes(x="cases_avg", y = "deaths_avg", color='week')) + geom_line(aes(x="cases_avg", y = "a0.001")) + geom_line(aes(x="cases_avg", y = "a0.01")) 


# In[105]:


ggplot(data) + geom_point(aes(x="cases_avg", y = "deaths_avg", color='week')) + geom_line(aes(x="cases_avg", y = "a0.1")) + geom_line(aes(x="cases_avg", y = "a1"))


# Using just your eyes, which straight line seems to best fit the data? What does best fit mean? 
# The line with the slope 0.01 is the best fit for the given data since it is passing through the plotted data points.

# # Problem 4 Github
# 
# - Create a github repository called "Che_Math".
# - Create a read me file that explain that this repository will include code related to homeworks and projects from this course.
# - Push this homework assignment to your repository.
# - Enter the link you your repository here:
# 
# 
# **Upload your jupyter notebook to Canvas for grading. Thank you!

# In[ ]:




