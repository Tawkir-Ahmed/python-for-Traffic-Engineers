"""
# Book: SciPy and NumPy: Optimizing & Boosting Your Python Programming
#Ch: 02; Numpy
Code: https://github.com/ebressert/ScipyNumpy_book_examples/tree/master/python_examples
"""
import numpy as np

#create an array with 10^7 elemnts.
arr = np.arange(1e7)

#converting ndarray to list
larr = arr.tolist()

#Lists cannot by default broadcast,
#so a funciton is cooded to emulate
# waht an ndarray can do

def list_times(alist, scalar):
    for i, val in enumerate(alist):
        alist[i] = val * scalar
    return alist

# Using IPython's magic timeit command
%%timeit 
arr * 1.1

%%timeit
list_times(larr, 1.1)

#ndarray is much faster than Python loop
# Matrix object vs ndobject

import numpy as np
#crating a 3D numpy array
arr = np.zeros((3, 3, 3))

# Tryign to convert array to a matrix, which will not work
mat = np.matrix(arr) # "ValueErros: shae too large to be a matrix."

# Array Creating oand data Typing
#First we create a list and then 
#Wrap it with the np.array() funciton.
alist = [1, 2, 3]
arr = np.array(alist)

#Creating an array of zeros with five elements
arr = np.zeros(5)

# Waht if we want to create an array going from  0 to 100?
arr = np.arange(100)

# Or 10 to 100?
arr = np.arange(10, 100)

#If you want 100 steps from 0 to 1....
arr = np.linspace(0, 1, 100)

# Or if you want to generate an array form 1 to 10
# in log10 space in 100 steps..
arr = np.logspace(0, 1, 100, base=10.0)

#Creating a 5X5 array of zeros (an image)
image = np.zeros((5,5))

#Creating a 5X5X5 cube of 1's
#The astype() method sets the array with interger elements.
cube = np.zeros((5, 5, 5)).astype(int) + 1

#Or even simpler with 16-bit floating-pont precision...
cube = np.ones((5, 5, 5)).astype(np.float16)

# you can change the datatype 
#Array of zero integers
arr = np.zeros(2, dtype=int)

#Array of zero floats
arr = np.zeros(2, dtype=np.float32)

#reshpe the array in many other way
# if you have 25 element array, you can make it a 5X5 array and 3-dimentional

#Creating an array with elements from 0 to 999
arr1d = np.arange(1000)

#Now reshapingthe arry to a 10X10X10 3D array
arr3d = arr1d.reshape((10, 10, 10))

#The reshape commadn can alternatively be called this way
arr3d = np.reshape(arr1d, (10, 10, 10))

#Inversely, we can flatten arrays
arr4d = np.zeros((10, 10, 10, 10))
arr1d = arr4d.ravel()

print(arr1d.shape)

#Record Arrays
#Creating an array of zeros and defining column types
recarr = np.zeros((2,), dtype=('i4, f4, a10'))
toadd = [(1, 2., 'Hello'), (2, 3., 'World')]
recarr[:] = toadd

#Creating an array of zeros and defining column types
recarr = np.zeros((2,), dtype=('i4, f4, a10'))

#Now creating the columns we want to put
#in the recarry
col1 = np.arange(2) + 1 #+1 is added to each element in the NumPy array created by np.arange(2)
col2 = np.arange(2, dtype=np.float32)
col3 = ['Hello', 'World']

#Here we create a list of tuples that is
#Identical to the previous toadd list.
toadd = zip(col1, col2, col3) #The zip() function is used to combine multiple iterables (such as lists or arrays) element-wise into a single iterable of tuples. 

# Convert to a NumPy array with the same dtype as recarr
toadd = np.array(list(toadd), dtype=recarr.dtype)
#Assigning vlues to recarr
recarr[:] = toadd

#Assigning smaes to each column, which
#are now by defalut called 'f0', 'f1', and 'f2'.
recarr.dtype.names = ('Integers', 'Floats', 'Strings')

#If we want to assess one of the columns by its name, we
#can do the following.

recarr('Integers') # error
#arr([1, 2], dtype=int32)

#Indexing and Slicing
alist = [[1, 2], [3, 4]]

#To return the (0,1) element we msut index as shown below.
alist[0][1]

#Converting the list defined above into an array
arr = np.array(alist)

#To retun the (0, 1) elemetn we use ...
arr[:,1]

#Accessing the columns is achieved in the same way,
#Which is the bottom row.
arr[1,:]

#Creating an array
arr = np.arange(5)

#Creating the index array
index = np.where(arr > 2)
print(index)

#Creating the desired array
new_arr = arr[index]

#We use the previous array
new_arr = np.delete(arr, index)

index = arr > 2
print = arr > 2

new_arr = arr[index]

#Bollean Satements and NumPy Arrays: logical statement
#Crating an image
img1 = np.zeros((20, 20)) + 3
img1[4:-4, 4:-4] = 6
img1[7:-7, 7:-7] = 9
#See Plot A

#Let's filter out all values larger than 2 and less than 6.
index1 = img1 > 2
index2 = img1 < 6
compound_index = index1 & index2

#The compound statemetn can alternatively be written as
compound_index = (img1 >3) & (img1 < 7)
img2 = np.copy(img1)
img2[compound_index] = 0
#See Plot B.

#Makign the boolean arrays even more complex
index3 = img1 == 9
index4 = (index1 & index2) | index3
img3 = np.copy(img1)
img3[index4] = 0
#See Plot C.

import numpy as np
import numpy.random as rand

#Creating a 100-elemnt array with random values
#from a standara normal distribution or, in other
#words, a Gaussian distribtuion.
#The sigma is 1 and the mean is 0.
a = rand.randn(100)

#Here we geenrate an index fro filtering 
#out undersidrered elements.
index = a > 0.2
b = a[index]

#We execute some operation on the desired elements.
b = b ** 2 - 2

#Then we put the modified elements baxck into the
#original array.
a[index] = b

#Page 12
#Read and Write
#Text Files

#Opening the text file with the 'r' option,
#Which only allows reding capability
f = open('somefile.txt', 'r')

#Parsing the file and splitting each line,
#Which creates a list where each element of
#it is one line
alist = f.readlines()

#Closing file
f.close()

#After a few operations, we open a new text file
#to write the data with the 'w' option. If there
#was data already existing in the file, it will be overwritten.
f = open('newtextfile.txt', 'w')

#Writing data to file
f.writelines(newdata)

#Closing file
f.close()

#Math
import numpy as np

#Defining the matrices
A = np.matrix([[3, 6, -5],
               [1, -3, 2],
               [5, -1, 4]])

B = np.matrix([[12],
               [-2],
               [10]])

#Solving for the variables, where we invert A
X = A ** (-1) * B

print(X)

#matrix([[1.75],
# [1.75],
# [0.75]])

import numpy as np
import pandas as pd
import math
aa = np.array([[3, 6, -5],
              [1, -3, 2],
              [5, -1, 4]])
#Define the array
bb = np.array([12, -2, 10])

#Solving for the variables, where we invert A
x = np.linalg.inv(aa).dot(bb)
print(x)

#array([1.75, 1.75, 0.75])

"""
Chap: SciPy: 03
this is the optimization and minimization package
in code: see 311
"""
#Optimization and Minimization
#Data Modeling and Fitting
#curve_fit
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Let's create a function to model and create data
def func(x, a, b):
    return a * x + b

# Generating clean data
x = np.linspace(0, 10, 100)
y = func(x, 1, 2)

# Adding noise to the data
yn = y + 0.9 * np.random.normal(size=len(x))

# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)

#popt returns the best fit values for parameters of the given model (func)
print(popt)

# Plot out the current state of the data and model
ym = func(x, popt[0], popt[1])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Function')
ax.scatter(x, yn)
ax.plot(x, ym, c='r', label='Best fit')
ax.legend(loc='upper left')
fig.savefig('scipy_311_ex1.pdf', bbox_inches='tight')
plt.show()

'''
equation:
a*exp((-(x-mean)**2)/(2*var**2))
see: 311ex github
'''
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Let's create a function to model and create data
def func(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

# Generating clean data
x = np.linspace(0, 10, 100)
y = func(x, 1, 5, 2)

# Adding noise to the data
yn = y + 0.2 * np.random.normal(size=len(x))

# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)

#print("Optimized Parameters:", popt)
#print("Covariance Matrix:", pcov)
#popt returns the best fit values for parameters of the given model (func)
#print(popt) # not collable of numpy.ndarray

# Plot out the current state of the data and model
ym = func(x, popt[3], popt[4], popt[5]) # i just input the value from popt array
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Function')
ax.scatter(x, yn)
ax.plot(x, ym, c='r', label='Best fit')
ax.legend(loc='upper left')
fig.savefig('scipy_311_ex2.pdf', bbox_inches='tight')
plt.show()

#pg: 20

"""two gaussian model define in a equation"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Two-Gaussian model
def func(x, a0, b0, c0, a1, b1, c1):
    return a0*np.exp(-(x - b0) ** 2/(2 * c0 ** 2))\
    + a1 * np.exp(-(x - b1) ** 2/(2 * c1 ** 2))

#Generating clean data
x = np.linspace(0, 20, 2000)
y = func(x, 1, 3, 1, -2, 15, 0.5)

#Adding noise to the data
yn = y + 0.2 * np.random.normal(size=len(x))

#Since we are fitting a more complex fucntion,
#providing guesses for the fitting will lead to
#better results.
#guesses = [1, 3, 1, 1, 15, 1]
#Executing curve_fit on noisy data
#popt, pcov = curve_fit(func, x, yn, p0=guesses)

# Plot out the current state of the data and model
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Function')
ax.scatter(x, yn)

#Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn, p0= [1, 3, 1, 1, 15, 1]) # p0 for initial guess of values
# curve_fit function aims to find the best parameters for both Gaussians to fit the data, and the resulting popt contains optimized parameters for both Gaussians.

#popt returns the best fit values for parameters of the given model (func)
print(popt)

ym = func(x, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]) #The best-fit curve (ym) represents the combination of two Gaussian functions that fit the noisy data.
ax.plot(x, ym, c='r', label='Best fit')
ax.legend(loc='upper left')
fig.savefig('scipy_311_ex3.pdf', bbox_inches='tight')
plt.show()

#if i want to plot he two equation seperately
# Plot out the current state of the data and model
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Function')
ax.scatter(x, yn)

# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn, p0=[1, 3, 1, 1, 15, 1])  # p0 for initial guess of values

# Plot the individual Gaussian components
gaussian_1 = func(x, popt[0], popt[1], popt[2], 0, 0, 0)
gaussian_2 = func(x, 0, 0, 0, popt[3], popt[4], popt[5])
ax.plot(x, gaussian_1, c='b', label='Gaussian 1')
ax.plot(x, gaussian_2, c='g', label='Gaussian 2')

ax.legend(loc='upper left')
fig.savefig('scipy_311_ex3.pdf', bbox_inches='tight')
plt.show()

"""
VVI: solution to functions
**************************
""" 
from scipy.optimize import fsolve
import numpy as np

line = lambda x: x + 3 #y = x+3 if y=0, x=-3

solution = fsolve(line, -2)
print(solution)
# Plotting output

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
x = np.linspace(-5, 5, 1000)
y = line(x)
ax.hlines(0, -5, 0, color='black', alpha=0.15, linestyle='--')
ax.plot(x, y, label='Function')
ax.scatter(solution[0], 0, marker='o',\
           edgecolor='red', facecolor='none', s=100, label='Root')
ax.legend(loc='upper left')
ax.set_xlim(-4.5, -1.5)
ax.set_ylim(-1.5, 1.5)
fig.savefig('scipy_312_ex1.pdf', bbox_inches='tight')
plt.show()

"""Finding the intersection points between two equations is nearly as simple
source: sipy_312_ex2.py
"""

from scipy.optimize import fsolve
import numpy as np

#Defining function to simplify interasection solution
def findIntersection(func1, func2, x0):
    return fsolve(lambda x: func1(x) - func2(x), x0)

#Defining fucntions that will intersect
funky = lambda x : np.cos(x / 5) * np.sin(x / 2)
line = lambda x : 0.01 * x - 0.5

#Defining range and getting solutions on intersection points
x = np.linspace(0, 45, 10000)
result = findIntersection(funky, line, [15, 20, 30, 35, 40, 45])

#Printing out results for x and y
print(result, line(result))
#Pg 22

#plotting output
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot(x, funky(x), label='Funky func')
ax.plot(x, line(x), label='Line func')
ax.scatter(result, line(result), marker='0',
           edgecolor='red', facecolor= 'none', s=100)
ax.legend(loc='lower left')
ax.set_xlilm(0, 45)
ax.set_ylim(-1, 1)
fig.savefig('scipy_312_ex2.pdf', bbox_inches='tight')
plt.show()

"""Interpolation
source: scipy_32_ex1.py
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#setting up fake data
x = np.linspace(0, 10 * np.pi, 20)
y = np.cos(x)

#interpolating data
fl = interp1d(x, y, kind='linear')
fq = interp1d(x, y, kind='quadratic')

#x.min and x.max are used to make sure we do not
#go beyong the boundaries of the data for the 
#interpolation.
xint = np.linspace(x.min(), x.max(), 1000)
yintl = fl(xint)
yintq = fq(xint)

#plotting otuput
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot(xint, yintl, label='Linear')
ax.plot(xint, yintq, label='Quadratic')
ax.scatter(x, y, marker='0',
           edgecolor='red', facecolor= 'none', s=50)
ax.legend(loc='upper left')

ax.set_xlilm(0, 10 * np.pi)
ax.set_ylim(-2, 2)
fig.savefig('scipy_32_ex1.pdf', bbox_inches='tight')
plt.show()

#Interpolate noisy data
"""scipy_32_ex2.py"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

#setting up fake data with artificala noise
sample = 30
x = np.linspace(1, 10 * np.pi, sample)
y = np.cos(x) + np.log10(x) + np.random.randn(sample) / 10

#Interpolating the data
f = UnivariateSpline(x, y, s=1) #s=smoothing factor

#x.min and x.max are used to make sure we do not 
#go beyond the boundaries of the data for the 
#interpolation.

xint = np.linspace(x.min(), x.max(), 1000)
yint = f(xint)
yclean = np.cos(xint) + np.log10(xint)

#making figure
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
ax.plot(xint, yint, label='Interpolation')
ax.plot(xint, yclean, label='Original', c='orange')
ax.scatter(x, y, marker='0',
           edgecolor='none', facevolor='black', s=20)
ax.legend(loc='upper left')

ax.set_xlim(1, 10 * np.pi)
fig.savefig('sicipy_32_ex2.pdf', bbox_inches='tight')
plt.show()

#multivariate to reprodeuce and image
""""scipy_32_ex3.py"""
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

#defining a function
ripple = lambda x, y: np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)

#generating gridded data. The complex number defines
#how many steps the grid data should have . without the
#complex number mgrid would only create a grid data structure
#with 5 steps.
grid_x, grid_y = np.mgrid[0:5:1000j, 0:5:1000j]

#generating sample that interpolation function will see
xy = np.random.rand(1000, 2)
sample = ripple(xy[:,0] * 5 , xy[:,1] * 5)

#interpolating data with a cubic
grid_z0 = griddata(xy * 5, sample, (grid_x, grid_y), method ='cubic')

#making figure.
fig = plt.figure(figsize=(8, 4))

x0, x1 = 0, 1000
y0, y1 = 0, 1000
ax1 = fig.add_subplot(121)
ax1.imshow(ripple(grid_x, grid_y).T, cmap=plt.cm.Blues,
           interpolation='nearest')
ax1.scatter(xy[:, 0]*1e3, xy[:,1]*1e3, facecolor='black',
            edgecolor='none', s=1)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim(x0, x1)
ax.set_ylim(y0, y1)

ax2 = fig.add_subplot(122)
ax2.imshow(grid_z0.T, camp=plt.cm.Blues, interpolation='nearest',
           vmin=0.05, vmax=7.87)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)

fig.savefig('scipy_32_ex3.pdf', bbox_inches='tight')
plt.show()


#SmoothBivariateSpline
"""scipy_32_ex4.py"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline as SBS

#Defining a function
ripple = lambda x, y: np.sqrt(x ** 2 + y ** 2) + np.sin(x ** 2 + y ** 2)

#Generating gridded data
grid_x, grid_y = np.mgrid[0:5:1000j, 0:5:1000j]

#Generating saplte that interpolation funciton will see
xy = np.random.rand(1000, 2)
x, y = xy[:, 0], xy[:, 1]
saplte = ripple(xy[:, 0] * 5, xy[:, 1] * 5)

#Interpolating data
fit = SBS(x * 5, y * 5, saplte, s=0, kx=4, ky=4)
interp = fit(np.linspace(0, 5, 10000), np.linspace(0, 5, 1000))

#Making figure
fig = plt.figure(figsize=(8, 4))

x0, x1= 0, 1000
y0, y1= 0, 1000
ax1 = fig.add_subplot(121)
ax1.imshow(ripple(grid_x, grid_y).T, cmap=plt.cm.Blues, interpolation='nearest')
ax1.scatter(xy[:, 0] * 1e3, xy[:, 1] * 1e3, facecolor='black', edgecolor='none', s=1)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim(x0, x1)
ax1.set_ylim(y0, y1)

ax2 = fig.add_subplot(121)
ax2.imshow(interp.T, cmap=plt.cm.Blues, interpolation='nearest', vmin=0.05, vmax=7.87)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_xlim(x0, x1)
ax2.set_ylim(y0, y1)

plt.show()
save_path = "Downloads\scipy_32_ex4.pdf"
fig.savefig(save_path, bbox_inches='tight')

"""Integration"""
"""quad: for integrate result"""
"""scipy_331_ex1.py"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

#Defining function to integrate
func = lambda x: np.cos(np.exp(x)) ** 2

#Integratign fucntion with upper and lower
#Limits of 0 and 3, respectively
solution = quad(func, 0, 3) #quad:  to perform numerical integration
print(solution)

#The first element is the desired value
#and the second is the error
#(1.296467785724373, 1.397797186265988e-09)

#Plotting otuput
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
x1 = np.linspace(-1, 4, 10000)
x2 = np.linspace(0, 3, 10000)
ax.plt(x1, func(x1), label='Function')
ax.fill_between(x2, 0, func(x2), alpha=0.2)
ax.set_xlim(-1, 4)
ax.set_ylim(0, 1.05)
plt.show()

save_path = "Downloads\scipy_331_ex1.pdf"
fig.savefig(save_path, bbox_inches='tight')

"""scipy_311_ex2.py"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Let's create a funciton to model and create data
def func(x, a, b, c):
    return a * np.exp(-(x -b) ** 2 / (2 * c ** 2))

#Generatingg clean data
x = np.linspace(0, 10, 100)
y = func(x, 1, 5, 2)

#Adding noise to the data
yn = y + 0.2 * np.random.normal(size=len(x))

#Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)

#popt returns the best fit values for parameters fo the given model (function)
print(popt)

#Plot otut eh current state of the dta and model
ym = func(x, popt[0], popt[1], popt[2])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, c='k', label='Funciton')
ax.sxatter(x, yn)
ax.plot(x, ym, c='r', label='Best fit')
ax.legend(loc='upper left')
plt.show()

fig.savefig('Downloads\scipy_331_ex2.pdf', bbox_inches= 'tight')

"""scipyt_332_ex1.py"""
import numpy as np
from scipy.integrate import quad, trapz
import matplotlib.pyplot as plt

#Setting up fake data.
x = np.sort(np.random.randn(150) * 4 +4).clip(0, 5) # clip: limit the x value within the range 0-5
# x = np.clip()
func = lambda x: np.sin(x) * np.cos(x ** 2) + 1
y = func(x)

#integratig fucntion with upper and lower
#Limits of 0 and 5, respectively.
fsolution = quad(func, 0, 5)
dsolution = trapz(y, x=x)         #trapz: trapezoidal integration
print('fsolution = ' + str(fsolution[0]))
print('dsoution =' + str(dsolution))
print('The difference is' + str(np.abs(fsolution[0] - dsolution)))

fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(111)
x1 = np.linspace(-1, 6, 10000)
x2 = np.linspace(0, 5, 10000)

ax.plot(x1, func(x1), label='Funciton')
ax.fill_between(x2, 0, func(x2), alpha=0.2)
ax.scatter(x, y, marker='o', edgecolor='none', facecolor='red', s=7, zorder=3, label='Data points')
ax.legend(loc='upper left')
ax.set_xlim(-1, 6)
ax.set_ylim(0, 2.5)
plt.show()

fig.savefig('Downloads\scipy_332_ex1.pdf', bbox_inches= 'tight')

"""3.4 Statistics"""
''''Difference between numpy and scipy
numpy: basic statistical funciton: std, median, argmax, argmin
scipy: extended collection of statistical tools: distributions (continuous or discrete) and functions'''
import numpy as np

#Constructing a random array with 1000 elements
x = np.random.randn(1000)

#Calculating several of the built-in methods
#that numpy.array has
mean = x.mean()
std = x.std()
var = x.var()

#Continuous and Discrete Distributions
"""scipy_341_ex1.py"""
#Probability density functions (PDFs)
#Cumulative distribution funciton (CDFs)
import subprocess
import numpy as np
import scipy.stats as s
import matplotlib.pyplot as plt

# Setting up figure size and spacing
fig = plt.figure(figsize=(8, 9))
fig.subplots_adjust(hspace=0, wspace=0)

# The distributions that will be plotted
dists = ['norm', 'lognorm', 'gamma', 'invgauss', 'cauchy',
         'logistic', 'maxwell', 'powerlaw', 'rdist',
         'wald', 'alpha', 'rayleigh', 'triang', 'lomax',
         'laplace', 'gilbrat', 'fisk', 'erlang', 't', 'nakagami']

dists = np.sort(dists)

for i, D in enumerate(dists):
    print(D)
    x = np.linspace(-5, 5, 1000)
    func = s.__dict__[D]

    try:
        dist = func(loc=0)
        pdf = dist.pdf(x)
    except:
        dist = func(0.7)
        pdf = dist.pdf(x)

    ax = fig.add_subplot(5, 4, i + 1)
    D = D.capitalize()  # Capitalize the first letter of the distribution name
    ax.plot(x, pdf, label=D)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_xlim(-4, 4)  # Corrected method name
    ax.set_ylim(0, pdf.max() * 1.5)  # Corrected method name
    ax.legend(loc='upper right', markerscale=0.0001, prop={'size': 10})

    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    llines = leg.get_lines()
    frame = leg.get_frame()
    plt.setp(llines, linewidth=0)
    leg.draw_frame(False)

savename = 'scipy_341_ex1.pdf'
plt.show()
plt.savefig(savename, bbox_inches='tight')  # Corrected saving method

# Calling OS for pdfcrop (Unix and Linux based systems only)
try:
    subprocess.call(['pdfcrop', savename, savename])
except Exception as e:
    print("Error: PDF cropping failed. Ensure pdfcrop is available.")
    print(e)


"""scipy_341_ex2.py"""
import numpy as np
from scipy.stats import norm

#Setup the sample range
x = np.linspace(-5, 5, 1000)

#Here set up the parameters for the normal distribution.
#Where loc is the mean and scale is the standard deviation.
dist = norm(loc=0, scale=1)

#Calling norm's PDF and CDF
pdf = dist.pdf(x)
cdf = dist.cdf(x)

#Here we draw out 500 random values from
sample = dist.rvs(500)


"""scipy_341_ex3.py"""
import numpy as np
from scipy.stats import geom

#Here set up the prameters for the normal distribution.
#Where loc is the mean and scale is the standard deviation.
p = 0.5
dist = geom(p) #geom: geometrical distribution

#Setup the sample range
x = np.linspace(0, 5, 1000)

#Calling norm's PMF and CDF
pmf = dist.pmf(x)
cdf = dist.cdf(x)

#Here we draw out 500 random values from
sample = dist.rvs(500)

#Functions: 60 statistical functions: Kolmogorov-Smirnov, kstest, normaltest
"""scipy_342_ex1"""
import numpy as np
from scipy import stats

#Generating a normal distribution sample
#with 100 elements
sample = np.random.randn(100)

#The harmonic mean: Sample values have to
#be greater than 0.
out = stats.tmean(sample, limits = (-1, 1))
print('\nTrimmed mean = ' + str(out))

#Calculating the skewness fo the sample
out = stats.skew(sample)
print('\nSkewness= ' + str(out))

#Additionsally, there is a handly summary funciton called
#describe, which gives a quick look at the data.
out = stats.describe(sample)
print('\nSize = ' + str(out[0]))
print('Min = ' + str(out[1][0]))
print('Max = ' + str(out[1][1]))
print('Mean = ' + str(out[2]))
print('Variance = ' + str(out[3]))
print('Skewness = ' + str(out[4]))
print('Kurtosis = ' + str(out[5]))

"""scipy_342_ex2"""
import numpy as np
from scipy import stats

#Generating a normal distribution sample
#with 100 lements
sample = np.random.randn(100)

#The harmonic mean: Sample values have to
#be greater than 0.
outh = stats.hmean(sample[sample > 0])        #Harmonic Mean (hmean):

#The mean, where values velow -1 and above 1 are
#removed for the mean calculation
outt = stats.tmean(sample, limits=(-1, 1))    #Truncated Mean (tmean)

#Calculating the skewness of thee sample
outs = stats.skew(sample)

outd = stats.describe(sample)

#3.5 spatial and clustering analysis
#Vector Quantization
"""scipy_351_ex1.py"""
import numpy as np
from scipy.cluster import vq
import matplotlib.pyplot as plt

#Creating data
c1 = np.random.randn(100, 2) + 5
c2 = np.random.randn(30, 2) - 5
c3 = np.random.randn(50, 2)

#Pooling all teh data into one 150 X 2 array
data = np.vstack([c1, c2, c3])

#Calculating the cluster centriods and variance
#from kmens
centroids, variance = vq.kmeans(data, 3)

#The idenfified varaible contains the information
#we need to separate the points in clusters
#based on the vq fucniton.
identified, distance = vq.vq(data, centroids)

#Retrieving coordinates form points in each vq
#identified core
vqc1 = data[identified == 0]
vqc2 = data[identified == 1]
vqc3 = data[identified == 2]

#Setting up  plot details
x1, x2 = -10, 10   # variable assignment: to set initial value
y1, y2 = -10, 10

fig = plt.figure()
fig.subplots_adjust(hspace=0.1, wspace=0.1)

ax1 = fig.add_subplot(121, aspect='equal')
ax1.scatter(c1[:, 0], c1[:, 1], lw=0.5, color='#00CC00')
ax1.scatter(c2[:, 0], c2[:, 1], lw=0.5, color='#028E9B')
ax1.scatter(c3[:, 0], c3[:, 1], lw=0.5, color='#FF7800')
ax1.xaxis.set_visible(False)
ax1.set_ylim(y1, y2)
ax1.text(-9, 8, 'Original')

ax2 = fig.add_subplot(122, aspect='equal')
ax2.scatter(vqc1[:, 0], vqc1[:, 1], lw=0.5, color='#00CC00')
ax2.scatter(vqc2[:, 0], vqc2[:, 1], lw=0.5, color='#028E9B')
ax2.scatter(vqc3[:, 0], vqc3[:, 1], lw=0.5, color='#FF7800')
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_xlim(x1, x2)
ax2.set_ylim(y1, y2)
ax2.text(-9, 8, 'VQ idenfified')

plt.show()
fig.savefig('Downloads\scipy_351_ex1.pdf', bbox_inches='tight')

#Hierarchical Clustering
"""scipy_352_ex1.py"""

import numpy as np
import matplotlib.pyplot as plt  # Corrected alias name
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hy

# Creating a cluster of clusters function
def clusters(number=20, cnumber=5, csize=10):
    # Note that the way the clusters are positioned is Gaussian randomness.
    rnum = np.random.rand(cnumber, 2)
    rn = rnum[:, 0] * number
    rn = rn.astype(int)
    rn[np.where(rn < 5)] = 5
    rn[np.where(rn > number / 2.)] = round(number / 2., 0)
    ra = rnum[:, 1] * 2.9
    ra[np.where(ra < 1.5)] = 1.5

    cls = np.random.randn(number, 3) * csize

    # Random multipliers for the central point of clusters
    rxyz = np.random.randn(cnumber - 1, 3)
    for i in range(cnumber - 1):
        tmp = np.random.randn(rn[i + 1], 3)  # Corrected typo
        x = tmp[:, 0] + (rxyz[i, 0] * csize)
        y = tmp[:, 1] + (rxyz[i, 1] * csize)
        z = tmp[:, 2] + (rxyz[i, 2] * csize)
        tmp = np.column_stack([x, y, z])
        cls = np.vstack([cls, tmp])
    return cls

# Generate a cluster of clusters and distance matrix
cls = clusters()
D = pdist(cls[:, 0:2])
D = squareform(D)

# Compute and plot the first dendrogram
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
Y1 = hy.linkage(D, method='complete')
cutoff = 0.3 * np.max(Y1[:, 2])
Z1 = hy.dendrogram(Y1, orientation='right', color_threshold=cutoff)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)

# Compute and plot the second dendrogram
ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
Y2 = hy.linkage(D, method='average')
cutoff = 0.3 * np.max(Y2[:, 2])
Z2 = hy.dendrogram(Y2, color_threshold=cutoff)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

# Plot the distance matrix
ax3 = fig.add_axes([0.3, 0.1, 0.6, 0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1, :]
D = D[:, idx2]
ax3.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.YlGnBu)
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)

# Plot colorbar
plt.show()
fig.savefig('Downloads\\scipy_352_ex1.pdf', bbox_inches='tight')  # Corrected file path

## Hierarchically Cluster
import numpy as np
import matplotlib.pyplot as plt  # Corrected alias name
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as hy
#Same imports and cluster function from the previous example
#follow through here.

#Here we define a function to collect the coordinates of
#each point of the different clusters.
def group(data, index):
    number = np.unique(index)
    groups = []
    for i in number:
        groups.append(data[index == i])

    return groups

#Creating a cluster of clusters
cls = clusters()

#Calculating the linkage matrix
Y = hy.linkage(cls[:, 0:2], method='complete')

#Here we use the fcluster function to pul out a
#collection of flat clusters from the hierarchical
#data structure. Note that we are using the same
#cutoff value as in the previous example for the dendrogram
#using the 'complete' method.
cutoff = 0.3 * np.max(Y[:, 2])
index = hy.fcluster(Y, cutoff, 'distance')

#Using the group function, we group points into their
#respective clusters.
groups = group(cls, index)

#Plotting clusters
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
colors = ['r', 'c', 'b', 'g', 'orange', 'k', 'y', 'gray']
for i, g in enumerate(groups):
    i = np.mod(i, len(colors))
    ax.scatter(g[:, 0], g[:, 1], c=colors[i], edgecolor='none', s=50)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
plt.show()

fig.savefig('Downloads\scipy_352_ex2.pdf', bbox_inches='tight')

#3.6 Signal and image Processing
"""scipy_36_ex1.py"""
import numpy as np
import imageio #Use imageio for image I/O
from glob import glob

#Getting the list of files in the directory
files = glob('space/*.JPG')

#Opening up the first image for loop
im1 = imageio.imread(files[0]).astype(np.float32)

#Starting loop and continue co-adding new images
for i in range(1, len(files)):
    print(i)
    im1 += imageio.imread(files[i]).astype(np.float32)

#Savign img
imageio.imsave('Downloads\scipy_36_ex1.jpg', im1) #error in the imcode

#Image
"""scipy_36_ex2.py"""
import numpy as np
import imageio #Use imageio for image I/O
from scipy.misc import imread, imsave
from glob import glob

#This funciton allows us to place in the
#brightest pixels per x and y position between
#two images. It is similar to PIL's
#ImageChop.Lighter fucntion.
def chop_lighter(image1, image2):
    s1 = np.sum(image1, axis=2)
    s2 = np.sum(image2, axis=2)

    index = s1 < s2
    image1[index, 0] = image2[index, 0]
    image1[index, 1] = image2[index, 1]
    image1[index, 2] = image2[index, 2]
    return image1

#Getting the list of files in the directory
files = glob('space/*.JPG')

#Opening up the first image for looping
im1 = imread(files[0]).astype(np.float32)
im2 = np.copy(im1)

#Starting loop
for i in range(1, len(files)):
    print(i)
    im = imread(files[i]).astype(np.float32)
    #Same before
    im1 += im
    #im2 image shows star trails better
    im2 += chop_lighter(im2, im)

#Savign image with slight tweaking on the combination
#of the two images to show star trails with the
#co-added image.
imsave('Downloads\scipy_36_ex2.jpg', im1 / im1.max() + im2.max()*0.2)

#Chapter 04
#Linear Regression with 3d plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.datasets import make_regression

# Generating synthetic data for training and testing
X, y = make_regression(n_samples=100, n_features=2, n_informative=1, random_state=0, noise=50)

# Splitting the data into training and testing sets
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# Creating an instance of the linear regression model
regr = linear_model.LinearRegression()

# Training the model
regr.fit(X_train, y_train)

# Printing the coefficients
print(regr.coef_)  # [-10.25691752  90.5463984]

# Predicting y-values based on the training data
X1 = np.array([[1.2, 4]])
print(regr.predict(X1))  # [350.86036386]

# Calculating the R-squared score for the model
print(regr.score(X_test, y_test))

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')

# Plotting the training and testing data
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, c='#00CC00', label='Training Data', depthshade=False)
ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='#FF7800', label='Testing Data', depthshade=False)

# Creating a mesh grid for the regression line
x1, x2 = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))
z = regr.coef_[0] * x1 + regr.coef_[1] * x2

# Plotting the regression line
ax.plot_surface(x1, x2, z, alpha=0.1, color='k', label='Regression Line')

# Customize the plot
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.legend()
plt.show()
fig.savefig('Downloads/scikits_421_ex1.pdf', bbox_inches='tight')

#Clustering
#DBSCAN algorithm
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import DBSCAN

# Creating data
c1 = np.random.randn(100, 2) + 5
c2 = np.random.randn(50, 2)

# Creating a uniformly distributed background
u1 = np.random.uniform(low=-10, high=10, size=100)
u2 = np.random.uniform(low=-10, high=10, size=100)
c3 = np.column_stack([u1, u2])  # Corrected typo

# Pooling all the data into one 150 X 2 array
data = np.vstack([c1, c2, c3])

# Calculating the clusters with DBSCAN function.
# db.labels_ is an array with identifiers to the
# different clusters in the data.
db = DBSCAN(eps=0.95, min_samples=10).fit(data)  # Specify eps and min_samples
labels = db.labels_

# Retrieving coordinates for points in each
# identified core. There are two clusters
# denoted as 0 and 1 and the noise is denoted
# as -1. Here we split the data based on which
# component they belong to.
dbc1 = data[labels == 0]
dbc2 = data[labels == 1]
noise = data[labels == -1]

# Setting up plot details
x1, x2 = -12, 12
y1, y2 = -12, 12

fig = plt.figure()
fig.subplots_adjust(hspace=0.1, wspace=0.1)

ax1 = fig.add_subplot(121, aspect='equal')
ax1.scatter(c1[:, 0], c1[:, 1], lw=0.5, color='#00CC00')
ax1.scatter(c2[:, 0], c2[:, 1], lw=0.5, color='#028E9B')
ax1.scatter(c3[:, 0], c3[:, 1], lw=0.5, color='#FF7800')
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)
ax1.set_xlim(x1, x2)
ax1.set_ylim(y1, y2)
ax1.text(-11, 10, 'Original')

ax2 = fig.add_subplot(122, aspect='equal')
ax2.scatter(dbc1[:, 0], dbc1[:, 1], lw=0.5, color='#00CC00')
ax2.scatter(dbc2[:, 0], dbc2[:, 1], lw=0.5, color='#028E9B')  # Corrected color code
ax2.scatter(noise[:, 0], noise[:, 1], lw=0.5, color='#FF7800')
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)
ax2.set_xlim(x1, x2)
ax2.set_ylim(y1, y2)
ax2.text(-11, 10, 'DBSCAN identified')

fig.savefig('Downloads\\scikit_learn_clusters.pdf', bbox_inches='tight')  # Use forward slashes or double backslashes
plt.show()

#More tutorial about the 
#http://www.scikit-learn.org/stable/modules/linear_model.html
#http://www.scikit-learn.org/stable/modules/clustering.html


