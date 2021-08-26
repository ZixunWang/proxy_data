from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

a = np.random.randn(30*30*30)
a = a.reshape((30,-1))
print(a.shape)

embedded_900to2 = TSNE(n_components=2, method='exact').fit_transform(a)

embedded_900to100 = TSNE(n_components=100, method='exact').fit_transform(a)
embedded_100to2 = TSNE(n_components=2, method='exact').fit_transform(embedded_900to100)

print(embedded_900to2)
print(np.mean(embedded_900to2))

print(embedded_900to100)

print(embedded_100to2)

x = embedded_900to2[:,0]
y = embedded_900to2[:,1]
# indices = list(range(len(x)))
plt.scatter(np.array(x), np.array(y), s=1)


x = embedded_100to2[:,0]
y = embedded_100to2[:,1]
plt.scatter(np.array(x), np.array(y), s=1)
plt.savefig('tmp.png')