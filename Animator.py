# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:55:28 2016

@author: anie
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

filename = 'raw.npy'
movie_name = 'line.mp4'
class Animator:
    def __init__(self, filename):
        self.data = np.load(filename)
        print "data imported"
        self.fig = plt.figure('fig')
        self.frame = plt.scatter(self.data[:, 0, 0], self.data[:, 0, 1])
        # self.frame.set_array()
        self.n = len(self.data)
    # plot updating function    
    def update_plot(self, i):
        self.frame.set_offsets(self.data[:,i,:2])
        return self.frame,
    # build animation
    def animate(self):
        print "running animations"
        self.anim = animation.FuncAnimation(self.fig, self.update_plot, 
                                            frames=xrange(len(self.data[0])))
        
if __name__ == '__main__':
    movie = Animator(filename)
    movie.animate()
    print "saving movie"
    movie.anim.save(movie_name, fps=30, extra_args = ['-vcodec', 'libx264'])
    print "Done!"
    
    