# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 15:27:55 2025

@author: Ashlesha
"""
#17-4-25
'''WAP python to draw line charts of the
financial data of alphabet inc.
between october 3,2016 to october 7,2016'''
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('c:/Data-Science/2-Python_DS/fdata.csv')
df.plot()
plt.show()
############################################################################################################
#WAP Python plot 2 or more lines with legends
#different  widths and colors
#line 1 point
import matplotlib.pyplot as plt
x1=[10,20,30]
y1=[20,40,10]
#line 2 point
x2=[10,20,30]
y2=[40,10,30]
# set the x and y axis labels
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Two or more line  with different widths and color')
#display figure
plt.plot(x1,y1,color='blue',linewidth=3,label='line1-width-3')
plt.plot(x2,y2,color='red',linewidth=5,label='line2-width-5')
# Show legend
plt.legend()
# Display the plot
plt.show()
#################################################################################
#line 1 point
import matplotlib.pyplot as plt
x1=[10,20,30]
y1=[20,40,10]
#line 2 point
x2=[10,20,30]
y2=[40,10,30]
# set the x and y axis labels
plt.xlabel('x-axis')
plt.ylabel('y-axis')
#display figure
plt.plot(x1, y1,color='blue',linewidth=3,linestyle='dotted',label='Line 1 — width 3, dotted')
plt.plot(x2, y2,color='red',linewidth=5,linestyle='dashed',label='Line 2 — width 5, dashed')
plt.title('Two or more line  with different widths and color')
# Show legend
plt.legend()
# Display the plot
plt.show()
#################################################################################
#WAP Python plot 2 or more lines with legends
#lines and set the line markers
import matplotlib.pyplot as plt
# X–Y values
x = [1, 4, 5, 6, 7]
y = [2, 6, 3, 6, 3]
# Plotting with marker and custom style
plt.plot(x, y,color='red',linestyle='dashdot',linewidth=3,marker='o',markerfacecolor='blue',markersize=12)

# Set axis limits
plt.ylim(1, 8)
plt.xlim(1, 8)

# Label axes (fixed typos)
plt.xlabel('x-axis')
plt.ylabel('y-axis')

# Title
plt.title('Display marker')

# Show the plot
plt.show()

#################################################################################
#several line differnt format style in  one command using array
import numpy as np
import matplotlib.pyplot as plt
#sample time  at 200m intervals
t=np.arange(0.,5.,0.2)
#green dases ,blue square and red triangle
plt.plot(t,t,'g--',t,t**2,'bs',t,t**3,'r^')
'''
x=t,y=t
'g--'=green dased line--
plots a diagonal dashed green line (y=x)
t,t**2,'bs'
x=t,y=t**2(sqare)
'bs'=blue(b)square(s)as markers
plot t^2 as blue square

t,t**3,'r^'
x=t,y=t**3(cube)
'r^'=red triangel-up markers(^)
plot t3 as red triangel'''
plt.show()
#################################################################################
#use of plt.xticks
import matplotlib.pyplot as plt
x_pos=[0,1,2,3]
x=['Apple','BAnana','Mango','Orange']
plt.bar(x_pos,[10,15,7,12])
plt.xticks(x_pos,x)
plt.ylable('Quantity')
plt.title('Fruit stock')
plt.show()

#################################################################################
#WAP display bar chart of popularity of programming lang
import matplotlib.pyplot as plt
x=['java','python','php','css','js','html']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
'''
x_pos=[i for i,_in enumerate(x)]
this creact a list if index position for each langu
enumerate(x) gives(index,lang)pairs:
    (0,'java'),(1,'python'),...,(5,'c++')
    x_pos=[i for i, _ in enumerate(x)]extract just the indices:
        x_pos=[0,1,2,3,4,5]

'''
plt.bar(x_pos,popularity,color='blue')
plt.xlabel('Languages')
plt.ylabel('Popularity')
plt.title('Popularity of Programming Languages:')
plt.xticks(x_pos, x)
plt.show()

#################################################################################
#horizontal bar popularity of programming lang

import matplotlib.pyplot as plt
x=['java','python','php','css','js','html']
popularity=[22.2,17.6,8.8,8,7.7,6.7]
x_pos=[i for i, _ in enumerate(x)]
plt.barh(x_pos,popularity,color='green')
plt.xlabel('Popularity')
plt.ylabel('Languages')
plt.title('Popularity of Programming Languages:')
plt.yticks(x_pos, x)
plt.show()

#################################################################################
#create bar plot of scores by grup and gender used
# multiple x value on the same chat
import numpy as np
import matplotlib.pyplot as plt
n_groups=5
men_means=(22,30,33,30,26)
women_means=(25,32,30,35,29)
fig,ax=plt.subplots()
index=np.arange(n_groups)
bar_width=0.35
opacity=0.8
rects1 = plt.bar(index,men_means,bar_width,alpha=opacity,color='skyblue',label='Men')
rects2 = plt.bar(index + bar_width,women_means,bar_width,alpha=opacity,color='salmon',label='Women')
plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('Scores by Group and Gender')
plt.xticks(index + bar_width ,('G1','G2','G3','G4','G5'))
# Legend and layout
ax.legend()
plt.tight_layout()
# Display
plt.show()









