# Initial Belief
p = [0.2,0.2,0.2,0.2,0.2]
world = ['green', "red", "red", "green", "green"]
motion = [1,1]
measurements = ["red","red"]
#Weights
pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1

def sense(p,Z):
    q=[]
    for i in range (len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)

    for i in range (len(q)):
        q[i] = q[i]/s
    return q

def move(p,U):
    q=[]
    for i in range (len(p)):
        s = pExact * p[(i-U)%len(p)]
        s = s + pOvershoot * p[(i-U-1)%len(p)]
        s = s + pUndershoot * p[(i-U+1)%len(p)]
        q.append(s)
    return q



for i in range(len(measurements)):
    p = sense(p,measurements[i])
    print(p)
    p = move(p,motion[i])
    print(p)