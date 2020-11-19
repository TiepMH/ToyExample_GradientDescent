''' Tiep M. Hoang '''
###############################################################################

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################
''' Tensorflow '''
def loss_tf(x):
    k_th_batch = 0
    x1 = x[k_th_batch, 0]
    x2 = x[k_th_batch, 1]
    return tf.pow(x1,3) - 3*tf.multiply(x1, tf.pow(x2,2))

def grad_tf(x):
    return tf.gradients( loss_tf(x), x )
    
###############################################################################
''' Numpy '''
def loss(x):
    x1 = x[0][0]
    x2 = x[0][1]
    return x1**3 - 3*x1*(x2**2)

def find__x_next(x_current, grad_at__x_current):
    step_size = 0.05
    x = x_current - step_size * grad_at__x_current
    return x 

def Projected_Gradient_Descent(x, bounds):
    n_elements = np.shape(x)[1]
    for i in range(n_elements):
        xi = x[0][i]
        bound_xi = bounds[i]
        xmin = bound_xi[0]
        xmax = bound_xi[1]
        if xi > xmax:
            x[0][i] = xmax
        if xi < xmin:
            x[0][i] = xmin
    return x
    
###############################################################################
### CONSTRAINT: 
# bound_x1 = [min_of_x1, max_of_x1]
# bound_x2 = [min_of_x1, max_of_x1]
bound_x1 = np.array([-3,3])
bound_x2 = np.array([-3,3])
bounds = [bound_x1, bound_x2]

### Declare some tensorflow objects
x_tf = tf.placeholder(tf.float32, [None,2]) #x = [x1, x2]
#x_tf = tf.concat( [x1_tf, x2_tf], axis=1)
grad = grad_tf(x_tf) # grad = [df/dx1, df/dx2]

### Initialize the first value for x = [x1, x2]
x_current = np.array( [[1.5, -1]] )
loss_current = loss(x_current)

### Prepare for later use 
x_array = x_current
loss_array = loss_current 

''' Main Program '''
n_iterations = 100
epsilon = 10**(-5)
count_break = 0 
for k in range(n_iterations):
    ### Use tf to evaluate the gradient at x = x_current
    grad_at__x_current = tf.Session().run(grad, feed_dict={ x_tf: x_current })
    grad_at__x_current = grad_at__x_current[0]
    
    ### Find the next point
    x_next = find__x_next(x_current, grad_at__x_current)
    x_next = Projected_Gradient_Descent(x_next, bounds) 
    loss_next = loss(x_next)
    
    ## Check if the algorithm converges
    if np.abs( loss_current - loss_next ) < epsilon:
        count_break += 1
        break 
    else:
        # Update the next point iteratively 
        x_current = x_next 
        loss_current = loss_next 
    
    # x_current = x_next
    # loss_current = loss_next
    
    ### Store the results
    x_array = np.append(x_array, x_current, axis=0) 
    loss_array = np.append(loss_array, loss_current)
    
###############################################################################
''' Illustrate the gradient in 3D'''
fig = plt.figure(figsize=(8, 8))
ax = fig.gca(projection='3d')
ax.plot(x_array[:, 0], x_array[:, 1], loss_array, label='Gradient method', 
        color='r', marker='o', markersize=5, linestyle='-', linewidth=2)


###############################################################################
''' Illustrate the loss function in 3D '''
def graph_of_loss(x1,x2):
    # return (1-x1)**2 + 200*(x2 - x1**2)**2  # Rosenbrock's function
    return x1**3 - 3*x1*(x2**2) # Monkey saddle

bound_x1, bound_x2 = np.asarray(bounds)
x1min, x1max, x1_step = bound_x1[0], bound_x1[1], 0.05
x2min, x2max, x2_step = bound_x2[0], bound_x2[1], 0.05
x1 = np.arange(x1min, x1max, x1_step) 
x2 = np.arange(x1min, x2max, x2_step)

xx_1, xx_2 = np.meshgrid(x1, x2)
z = graph_of_loss(xx_1, xx_2) 
''' NOTE:
np.shape(z) = (45, 45). Do NOT write z = loss(x1, x2) because np.shape(z) = (1,45)
'''

#surf = ax.plot_surface(xx_1, xx_2, z, alpha=1)
surf = ax.plot_wireframe(xx_1, xx_2, z, alpha=1, label='Surface of $L(x_1,x_2)$',
                         linestyle='-', rstride = 5, cstride = 5)
#cset = ax.contourf(xx_1, xx_2, z, zdir='z', offset=np.min(z), cmap=cm.viridis)


ax.set_xlabel('$-3\leq x_1\leq 3$', fontsize=18)
ax.set_ylabel('$-3\leq X_2\leq 3$', fontsize=18)
ax.set_zlabel('Loss $L(x_1,X_2)$', fontsize=18)
ax.legend(loc='best', fontsize=12)
ax.set_title('Find $x=[x_1,x_2]$ so that $L(x_1, x_2) = x_1^3 - 3 x_1 x_2^2$ is minimized', fontsize=15)

###############################################################################
### Save figures
plt.savefig('RESULT__Example_2.jpeg', dpi=300, bbox_inches = 'tight')
