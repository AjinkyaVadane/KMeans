# resources used:
#       the slides provided by professor
import numpy as np
import matplotlib.pyplot as plot
x_input = []
for i in np.arange(-1, 10, .01):
    x_input.append(i)
dim = 0  # This one is for plotting 1D and 2D
h = [.1, 1, 5, 10]  #bandwidth
#Algortihm
# by referring to the algorithm in the ppt
def mykde(X, x, h):
    # X is the our data set
    # x is the current point passed as argument to the function
    # h is the bin size
    sumation = 0.0
    if (dim == 1):  # if 1D like part 2 and 3
        N = len(X)  # the number of point in the data set
        for i in range(len(X)):  # for each point in the data set,
            # calculate the current point - that point and then divide by the bin size
            u = (x - X[i]) / h
            if (abs(u) <= 0.5):  # if that value is within the hypercube
                k = 1
            else:
                k = 0
            sumation = sumation + k  # find the summation
        p = float(sumation / (N * h))  # find the kernal density
        return p, x

    if (dim == 2):  # if 2D like part 4
        # for 2D we will use h*h
        N = len(X)
        for i in range(len(X)):
            u1 = (x - X[i][0]) / h
            u2 = (x - X[i][1]) / h
            if (abs(u1) <= 0.5 and abs(u2) <= 0.5):
                k = 1
            else:
                k = 0
            sumation = sumation + k
        p = float(sumation / (N * h * h))
        return p, x
#part 3
mue1 = 5
mue2 = 0
sig1 = 1
sig2 = 0.2
N1 = 500
N2 = 500
X = np.concatenate((np.random.normal(mue1, sig1, N1), np.random.normal(mue2, sig2, N2)), 0)
dim = 1
p1 = [0.0] * (len(x_input))
p2 = [0.0] * (len(x_input))
p3 = [0.0] * (len(x_input))
p4 = [0.0] * (len(x_input))

for i in range(len(x_input)):  # for every point we calculate the kernel density using all 4 bin values
    p1[i], x_input[i] = mykde(X, x_input[i], h[0])
    p2[i], x_input[i] = mykde(X, x_input[i], h[1])
    p3[i], x_input[i] = mykde(X, x_input[i], h[2])
    p4[i], x_input[i] = mykde(X, x_input[i], h[3])

figure2, axes = plot.subplots(5, 1, constrained_layout=True)
figure2.canvas.set_window_title('Problem 2 - Part 3')

axes[0].hist(X, 100, density=True, color='cyan')
axes[1].plot(x_input, p1, c='cyan')
axes[2].plot(x_input, p2, c='cyan')
axes[3].plot(x_input, p3, c='cyan')
axes[4].plot(x_input, p4, c='cyan')

axes[1].set_title('h=.1 q2-part3')
axes[2].set_title('h=1 q2-part3')
axes[3].set_title('h=5 q2-part3')
axes[4].set_title('h=10 q2-part3')

#part 4
mue1 = [1, 0]
mue2 = [0, 1.5]
sig1 = [[0.9, 0.4], [0.4, 0.9]]
sig2 = [[0.9, 0.4], [0.4, 0.9]]
N = 500

# generate the data, 500 for each set
data = np.concatenate((np.random.multivariate_normal(mue1, sig1, N), np.random.multivariate_normal(mue2, sig2, N)), 0)

dim = 2  # 2-D

p1 = [0.0] * (len(x_input))
p2 = [0.0] * (len(x_input))
p3 = [0.0] * (len(x_input))
p4 = [0.0] * (len(x_input))

for i in range(len(x_input)):  #calculate the kernel density using all 4 bin values
    p1[i], x_input[i] = mykde(data, x_input[i], h[0])
    p2[i], x_input[i] = mykde(data, x_input[i], h[1])
    p3[i], x_input[i] = mykde(data, x_input[i], h[2])
    p4[i], x_input[i] = mykde(data, x_input[i], h[3])

figure3, axes = plot.subplots(5, 1, constrained_layout=True)
figure3.canvas.set_window_title('Problem 2 - Part 4')

axes[0].hist(data, 100, density=True)
axes[1].plot(x_input, p1, c='blue')
axes[2].plot(x_input, p2, c='blue')
axes[3].plot(x_input, p3, c='blue')
axes[4].plot(x_input, p4, c='blue')

axes[1].set_title('h=.1 q2-part4')
axes[2].set_title('h=1 q2-part4')
axes[3].set_title('h=5 q2-part4')
axes[4].set_title('h=10 q2-part4')


#part 2
m = 5
s = 1
N = 1000

X = np.random.normal(m, s, N)  # generate the 1000 data points

dim = 1

p1 = [0.0] * (len(x_input))
p2 = [0.0] * (len(x_input))
p3 = [0.0] * (len(x_input))
p4 = [0.0] * (len(x_input))

for i in range(len(x_input)):  #calculate the kernel density using all 4 bin values
    p1[i], x_input[i] = mykde(X, x_input[i], h[0])
    p2[i], x_input[i] = mykde(X, x_input[i], h[1])
    p3[i], x_input[i] = mykde(X, x_input[i], h[2])
    p4[i], x_input[i] = mykde(X, x_input[i], h[3])

figure1, axes = plot.subplots(nrows=5, ncols=1, constrained_layout=True)
figure1.canvas.set_window_title('Problem 2 - Part 2')

axes[0].hist(X, 100, density=True, color='g')
axes[1].plot(x_input, p1, c='g')
axes[2].plot(x_input, p2, c='g')
axes[3].plot(x_input, p3, c='g')
axes[4].plot(x_input, p4, c='g')

axes[1].set_title('h=.1 q2-part2')
axes[2].set_title('h=1 q2-part2')
axes[3].set_title('h=5 q2-part2')
axes[4].set_title('h=10 q2-part2')
######################################################################################3
# plot the histograms
plot.show()