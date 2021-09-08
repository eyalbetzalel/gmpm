import numpy as np
import matplotlib.pyplot as plt
x=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]]
y=[[1,2,3,4],[11,21,31,41],[11,12,13,14],[21,22,23,24]]
colours=['r','g','b','k']
label = ['a', 'b', 'c', 'd']

def plot_graph(title,epochs, metrics, labels,colors):

    plt.figure()

    for i in range(len(epochs)):
        plt.plot(epochs[i], metrics[i], c=colors(2*i), label=labels[i], marker='o')
    plt.title(title)
    plt.legend()
    plt.ylabel(title + " Score")
    plt.xlabel("Epoch")
    plt.savefig(title + "_.png")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, 2*n)


a = [1, 4, 2, 5]
b = ['aa', 'bb', 'cc', 'dd']

zipped_lists = zip(a, b)
sorted_pairs = sorted(zipped_lists)

tuples = zip(*sorted_pairs)
list1, list2 = [ list(tuple) for tuple in  tuples]

v=0

cmap = get_cmap(len(label))
plot_graph('title1',x,y,label,cmap)
plot_graph('title2',x,y,label,cmap)
