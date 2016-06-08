import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def main():
    # number of frames to run
    numframes = 100
    
    # number of points to initialize
    numpoints = 20
    
    # future position data of points
    position_data = np.random.random((numframes, numpoints, 2))
    
    # iinitialization of points (with color)
    x, y, c = np.random.random((3, numpoints))

    fig = plt.figure()
    # create scatter plot with initial colors
    scat = plt.scatter(x, y, c=c, s=100)

    # create animator
    ani = animation.FuncAnimation(fig, update_plot, frames=xrange(numframes),
                                  fargs=(position_data, scat))
    # save animation                                  
    ani.save('basic_example.mp4', fps=30, extra_args = ['-vcodec', 'libx264'])
    plt.show()

def update_plot(i, update_data, scat):
    
    scat.set_offsets(update_data[i])
    
    # change color data
#    scat.set_array(color_data[i])
    return scat,

main()