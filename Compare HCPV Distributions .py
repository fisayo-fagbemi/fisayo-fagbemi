#!/usr/bin/env python
# coding: utf-8

# In[68]:


import geostatspy.GSLIB as GSLIB          # GSLIB utilies, visualization and wrapper 
import geostatspy.geostats as geostats    # GSLIB methods convert to Python 


# In[69]:


import numpy as np                        # ndarrys for gridded data 
import pandas as pd                       # DataFrames for tabular data 
import os                                 # set working directory, run executables 
import matplotlib.pyplot as plt           # for plotting 
from scipy import stats                   # summary statistics 
import math                               # trig etc. 
import random 
from ipywidgets import interactive        # widgets and interactivity 
from ipywidgets import widgets                             
from ipywidgets import Layout 
from ipywidgets import Label 
from ipywidgets import VBox, HBox


# In[70]:


# interactive calculation of the random sample set (control of source parametric distribution and number of samples) 

l = widgets.Text(value=' Monte Carlo Simulation Demonstration, Modified from Michael Pyrcz, The University of Texas at Austin',layout=Layout(width='950px', height='30px'))  

operator = widgets.RadioButtons(options=['add', 'mult'],description='Operator:',disabled=False,layout=Layout(width='230px', height='50px')) 

 

L = widgets.IntSlider(min=1, max = 10000, value = 50, description = '$L$:',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

L.style.handle_color = 'gray'  

uiL = widgets.VBox([L,operator]) 

dist1 = widgets.Dropdown( 

    options=['Uniform','Triangular','Gaussian','Galton'], 

    value='Gaussian', 

    description='$X_1$:', 

    disabled=False, 

    layout=Layout(width='200px', height='30px') 

) 

min1 = widgets.FloatSlider(min=0.0, max = 100.0, value = 10.0, description = 'Min',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

min1.style.handle_color = 'blue' 

max1 = widgets.FloatSlider(min=0.0, max = 100.0, value = 30.0, description = 'Max',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

max1.style.handle_color = 'blue' 

ui1 = widgets.VBox([dist1,min1,max1],kwargs = {'justify_content':'center'})  

 

dist2 = widgets.Dropdown( 

    options=['Triangular', 'Uniform', 'Gaussian','Galton'], 

    value='Gaussian', 

    description='$X_2$:', 

    disabled=False, 

    layout=Layout(width='200px', height='30px') 

) 

min2 = widgets.FloatSlider(min=0.0, max = 100.0, value = 10.0, description = 'Min',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

min2.style.handle_color = 'red' 

max2 = widgets.FloatSlider(min=0.0, max = 100.0, value = 30.0, description = 'Max',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

max2.style.handle_color = 'red' 

ui2 = widgets.VBox([dist2,min2,max2],kwargs = {'justify_content':'center'}) 

 

dist3 = widgets.Dropdown( 

    options=['Triangular', 'Uniform', 'Gaussian','Galton'], 

    value='Gaussian', 

    description='$X_3$:', 

    disabled=False, 

    layout=Layout(width='200px', height='30px') 

) 

min3 = widgets.FloatSlider(min=0.0, max = 100.0, value = 10.0, description = 'Min',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

min3.style.handle_color = 'yellow' 

max3 = widgets.FloatSlider(min=0.0, max = 100.0, value = 30.0, description = 'Max',orientation='horizontal',layout=Layout(width='230px', height='50px'),continuous_update=False) 

max3.style.handle_color = 'yellow' 

ui3 = widgets.VBox([dist3,min3,max3],kwargs = {'justify_content':'center'}) 

 

ui = widgets.HBox([uiL,ui1,ui2,ui3]) 

ui2 = widgets.VBox([l,ui],) 

 

def make_dist(dist,zmin,zmax,L): 

    if dist == 'Triangular': 

        z = np.random.triangular(left=zmin, mode=(zmax+zmin)*0.5, right=zmax, size=L) 

        pdf = stats.triang.pdf(np.linspace(0.0,100.0,1000), loc = zmin, c = 0.5, scale = zmax-zmin)* 2 * L  

    if dist == 'Uniform': 

        z = np.random.uniform(low=zmin, high=zmax, size=L) 

        pdf = stats.uniform.pdf(np.linspace(0.0,100.0,1000), loc = zmin, scale = zmax-zmin) * 2 * L 

    if dist == 'Gaussian': 

        mean = (zmax + zmin)*0.5; sd = (zmax - zmin)/6.0 

        z = np.random.normal(loc = mean, scale = sd, size=L) 

        pdf = stats.norm.pdf(np.linspace(0.0,100.0,1000), loc = mean, scale = sd) * 2 * L 
        
        #amended to include Glaton
    if dist == 'Galton': 
        
        mean = (zmax + zmin)*0.5; sd = (zmax - zmin)/6.0
        
        std = sd * np.sqrt(1.0)
        
        z = np.random.lognormal(mean, sd, size=None)
        
        pdf = stats.lognorm.pdf(np.linspace(0.0,100.0,1000), mean, sd) * 2 * L
    
        
    return z, pdf 

         

def f_make(L,operator,dist1,min1,max1,dist2,min2,max2,dist3,min3,max3):  

    #np.random.seed(seed = 73073) 
    np.genfromtxt('hcpv.csv', delimiter=',')  #use values from HCPV

    x1, pdf1 = make_dist(dist1,min1,max1,L) 

    x2, pdf2 = make_dist(dist2,min2,max2,L) 

    x3, pdf3 = make_dist(dist3,min3,max3,L) 
    

    xvals = np.linspace(0.0,100.0,1000) 

    plt.subplot(241) 

    plt.hist(x1,density = False,bins=np.linspace(0,100,50),weights=None,color='blue',alpha=0.3,edgecolor='black') 

    plt.plot(xvals,pdf1,'--',color='black') 

    plt.xlim(0,100); plt.xlabel("$X_1$"); plt.title("First Feature, $X_1$"); plt.ylabel('Frequency') 

  

    plt.subplot(242) 

    plt.hist(x2,density = False,bins=np.linspace(0,100,50),weights=None,color='red',alpha=0.3,edgecolor='black') 

    plt.plot(xvals,pdf2,'--',color='black') 

    plt.xlim(0,100); plt.xlabel("$X_1$"); plt.title("Second Feature, $X_2$"); plt.ylabel('Frequency') 

  

    plt.subplot(243) 

    plt.hist(x3,density = False,bins=np.linspace(0,100,50),weights=None,color='yellow',alpha=0.3,edgecolor='black') 

    plt.plot(xvals,pdf3,'--',color='black') 

    plt.xlim(0,100); plt.xlabel("$X_1$"); plt.title("Third Feature, $X_3$"); plt.ylabel('Frequency') 

  

    y = np.zeros(L) 

    ymin = 0.0 

    if operator == "add": 

        y = x1 + x2 + x3 

    elif operator == "mult": 

        y = x1 * x2 * x3 

         

    ymax = max(round((np.max(y)+50)/100)*100,100) # round up to nearest hundreds to avoid the chart jumping around 

     

    plt.subplot(244) 

    plt.hist(y,density = False,bins=np.linspace(ymin,ymax,50),weights=None,color='black',alpha=0.5,edgecolor='black') 

    plt.xlabel("$Y$"); plt.title("Response Feature, $Y$"); plt.ylabel('Frequency') 

    plt.xlim(ymin,ymax) 

     

    plt.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=1.2, wspace=0.3, hspace=0.2) 

    plt.show()     

interactive_plot = widgets.interactive_output(f_make, {'L':L,'operator':operator,'dist1':dist1,'min1':min1,'max1':max1,'dist2':dist2,'min2':min2,'max2':max2,'dist3':dist3,'min3':min3,'max3':max3}) 

interactive_plot.clear_output(wait = True)                # reduce flickering by delaying plot updating 

 


# In[71]:


display(ui2, interactive_plot)                            # display the interactive plot 


# In[ ]:





# In[ ]:




