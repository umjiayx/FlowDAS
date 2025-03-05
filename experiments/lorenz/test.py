import torch


L = 10
W = 1

x = torch.zeros((L, 3))

xp = []

for i in range(L - W):
    p = x[i:i+W].reshape(-1)
    q = x[i+1:i+W+1].reshape(-1)
    print(p.shape, q.shape)
    comb = torch.cat((p, q), dim=0)
    xp.append(comb)

xp = torch.stack(xp)
print(xp.shape)



'''

x1 = torch.zeros((10, 3))
x2 = x1.reshape(-1)
print(x1.shape, x2.shape)

x3 = torch.cat((x2, x2), dim=0)
print(x3.shape)

'''


