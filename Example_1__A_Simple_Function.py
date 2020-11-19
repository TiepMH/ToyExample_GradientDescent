import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

###############################################################################
''' Tensorflow '''
def loss_tf(x):
    return tf.pow(x, 2) + tf.exp( -1 / tf.pow(10*(x-1), 2) )

def grad_tf(x):
    return tf.gradients( loss_tf(x), x )

''' Numpy '''
def find__x_next(x_current, grad_at__x_current):
    step_size = 0.01
    x = x_current - step_size * grad_at__x_current
    return x 

def loss(x):
    L = x**2 + np.exp( -1 / (10*(x-1))**2 )
    return L

def gradient_method__Nesterov(x_Nesterov, grad_at__x_Nesterov, k_th_iteration):
    x = find__x_next(x_Nesterov, grad_at__x_Nesterov)
    return x 

###############################################################################
''' Main Program '''
n_iterations = 40
epsilon = 10**(-5)

x_tf = tf.placeholder(tf.float32, [1,1])
grad = grad_tf(x_tf)

x_current = np.array([[1.5]])
loss_current = loss(x_current)

x_array = x_current
loss_array = loss_current 

count_break = 0 
for k in range(n_iterations):
    ### Use tf to evaluate the gradient at x = x_current
    grad_at__x_current = tf.Session().run(grad, feed_dict={ x_tf: x_current })
    grad_at__x_current = grad_at__x_current[0]
    
    ### Find the next point
    x_next = find__x_next(x_current, grad_at__x_current)
    loss_next = loss(x_next)
    
    ### Check if the algorithm converges
    if np.abs( loss_current - loss_next ) < epsilon:
        count_break += 1
        break 
    else:
        x_current = x_next
        loss_current = loss_next 
    
    ### Store the loss value in a list
    x_array = np.append(x_array, x_current) 
    loss_array = np.append(loss_array, loss_current)
   

''' Figures '''
plt.plot(x_array, loss_array, label='Gradient method', 
         marker='o', markersize=4, markerfacecolor='none')

interval_1 = [-0.11*i for i in range(15)]
interval_1.sort() 
interval_1 = np.vstack( interval_1 )
interval_2 = np.vstack( [0.11*(i+1) for i in range(15)] )
x_interval = np.reshape( np.append(interval_1, interval_2) , [30,1])
y_graph = loss(x_interval)
plt.plot(x_interval, y_graph, 
         label='Graph of $L(x) = x^2 + e^{-1 / (10*(x-1))^2 }$')

plt.xlabel('$-1.5\leq x \leq 1.5$', size = 12)
plt.ylabel('Loss  function  $L(x)$', size=12)
plt.legend(loc='upper left', fontsize=12)
plt.title('A Toy Example: Find $x$ so that $L(x)$ is minimized')

###############################################################################
### Save figures
plt.savefig('RESULT__Example_1.jpeg', dpi=300, bbox_inches = 'tight')

