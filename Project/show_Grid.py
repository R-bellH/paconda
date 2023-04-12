from matplotlib import pyplot
from matplotlib import colors
import imageio

colormap = colors.ListedColormap(["red", "green"])

frames = []


# def int(bool):
#     return 1 if bool else 0
def show_Grid(ghost_index):
    with open("grids_for_ghost_"+str(ghost_index)+".txt") as f:
        print "showing grid of ghost "+str(ghost_index)
        grid = []
        i=0
        for line in f.readlines():
            if line == "\n":
                if len(grid) > 0:
                    pyplot.figure(figsize=(10,5)) #len(grid[0]), len(grid)))
                    pyplot.imshow(grid, cmap=colormap)
                    pyplot.savefig('grid_img.png')
                    pyplot.savefig('./grid-animation/grid_img'+str(i)+'.png')
                    i+=1
                    image = imageio.imread('grid_img.png')
                    frames.append(image)
                    #pyplot.show()
                    grid = []
                continue
            str_line = line.split(' ')
            grid.append([int(c) for c in str_line])

        imageio.mimsave('./Grid_animation.gif',  # output gif
                        frames,  # array of input frames
                        fps=3)  # optional: frames per second


if __name__ == '__main__':
    show_Grid(1)