import math

#the network from figure 3 - code written w.r.t. this figure

#inputs and target
x = [1.,-1.]
target = 0

#first layer: w[i][j] goes from input i to hidden unit j
w = [[1.,1.,1.],[-1.,-1.,-1.]]
b = [0., 0., 0.]

#second layer: v[j][i] goes from hidden unit j to output i
v = [[1.,1.],[-1.,-1.],[-1.,-1.]]
c = [0., 0.]

def sigmoid(z):
    return 1.0/(1.0 + math.exp(-z))



#forward pass, part 1: hidden layer
#1 linear part: k[j] = sum over i of w[i][j] * x[i] + b[j], plus bias
k = [0.,0.,0.]
for j in range(3):
    for i in range(2):
        k[j] += w[i][j] * x[i]
    k[j] += b[j]
#2 sigmoid activation
h = [0.,0.,0.]
for j in range(3):
    h[j] = sigmoid(k[j])


#printing to see
print("k =", k)
print("h =", h)


#forward pass, part 2: outputs, softmax, loss
o = [0.,0.]
for i in range(2):
    for j in range(3):
        o[i] += h[j] * v[j][i]
    o[i] += c[i]

#softmax
e = [math.exp(o[0]), math.exp(o[1])]
total = e[0] + e[1]
y = [e[0]/total, e[1]/total]

#loss: minus log of true-class output
loss = -math.log(y[target])

print("o =", o)
print("y =", y)
print("loss =", loss)


#backward pass, part 1
do = [0.,0.]
for i in range(2):
    if i == target:
        do[i] = y[i] - 1
    else:
        do[i] = y[i]

#backward pass, part 2: second layer weights and biases
dv = [[0.,0.],[0.,0.],[0.,0.]]
for j in range(3):
    for i in range(2):
        dv[j][i] = h[j] * do[i]

dc = [0.,0.]
for i in range(2):
    dc[i] = do[i]

#3 hidden values: each h[j] feeds both outputs, so two terms are added
dh = [0.,0.,0.]
for j in range(3):
    for i in range(2):
        dh[j] += v[j][i] * do[i]

#4 back through sigmoid

dk = [0.,0.,0.]
for j in range(3):
    dk[j] = dh[j] * h[j] * (1 - h[j])

#5 first layer wieghts w and biases b
dw = [[0.,0.,0.],[0.,0.,0.]]
for i in range(2):
    for j in range(3):
        dw[i][j] = x[i] * dk[j]

db = [0.,0.,0.]
for j in range(3):
    db[j] = dk[j]

print("dw =", dw)
print("db =", db)
print("dv =", dv)
print("dc =", dc)