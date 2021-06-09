import matplotlib.pyplot as plot


def scatterplot3D(data, title, labels = ["x","y","z"], plt_components = (None,None,None)):
    # Unpack Plot components
    plt, fig, ax = plt_components

    # Create new plot if it is not passed as input
    if plt==None:
        fig = plot.figure()
        plt = plot
        ax = fig.add_subplot(111, projection='3d')

    # Set Axis Limit
    ax.set_xlim([min(data[:, 0]), max(data[:, 0])])
    ax.set_ylim([min(data[:, 1]), max(data[:, 1])])
    ax.set_zlim([min(data[:, 2]), max(data[:, 2])])

    # Set Plot labels
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    ax.set_zlabel(labels[2])
    #ax.axis('tight')

    # Plot data on axis
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=1)

    # Return new plot components tuple
    plt_components = (plt, fig, ax)
    return plt_components
    #plt.show()


def plot3D(data, title, labels = ["x","y","z"],color="black",plt_components = (None,None,None),line_width=1):
    # Unpack Plot components
    plt, fig, ax = plt_components

    # Create new plot if it is not passed as input
    if plt == None:
        fig = plot.figure()
        plt = plot
        ax = fig.add_subplot(111, projection='3d')

    # Set Axis Limit
    #ax.set_xlim([min(data[:, 0]), max(data[:, 0])])
    #ax.set_ylim([min(data[:, 1]), max(data[:, 1])])
#    ax.set_zlim([min(data[:, 2]), max(data[:, 2])])

    # Set Plot labels
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    ax.set_zlabel(labels[2])
    #ax.axis('tight')

    # Plot Patient_Data on Axis
    ax.plot(data[:, 0], data[:, 1], data[:, 2], color=color,linewidth=line_width)

    # Return new plot components tuple
    plt_components = (plt, fig, ax)
    return plt_components


def surfplot3D(data,title, labels = ["x","y","z"], clip_z=[None,None],plt_components = (None,None,None)):
    from matplotlib import cm

    # Unpack Surface Tuple
    surf_x, surf_y, surf_z = data

    # Unpack Plot components
    plt, fig, ax = plt_components

    # Create new plot if it is not passed as input
    if plt == None:
        fig = plot.figure()
        plt = plot
        ax = fig.add_subplot(111, projection='3d')

    #ax.set_xlim([min(surf_x), max(surf_x)])
    #ax.set_yli([min(surf_y), max(surf_y)])
    #ax.set_zlim([min(surf_z), max(surf_z)])

    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    ax.set_zlabel(labels[2])

    # clip data if enabled
    if clip_z[0] != None:
        max_z=clip_z[1]
        min_z=clip_z[1]
        surf_z[surf_z > max_z] = max_z
        surf_z[surf_z < min_z] = min_z

    surf = ax.plot_surface(surf_x, surf_y, surf_z, rstride=1, cstride=1, cmap=cm.jet, alpha=0.5)

    plt_components = (plt, fig, ax)
    return plt_components


def plot2D(data, title, labels = ["x","y"],color="black",linestyle='-',plt_components = (None,None,None)):
    # Unpack Plot components
    plt, fig, ax = plt_components

    # Create new plot if it is not passed as input
    if plt == None:
        fig = plot.figure()
        plt = plot
        ax = fig.add_subplot(111)

    # Set Axis Limit
    ax.set_xlim([min(data[:, 0]), max(data[:, 0])])
    ax.set_ylim([min(data[:, 1]), max(data[:, 1])])

    # Set Plot labels
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    # ax.axis('tight')

    # Plot Patient_Data on Axis
    ax.plot(data[:, 0], data[:, 1], color=color,linestyle=linestyle)

    # Return new plot components tuple
    plt_components = (plt, fig, ax)

    return plt_components


if __name__ == "__main__":
    from Modules import Old_Data_Handler

    d = Old_Data_Handler.placeholder_data
    plot3D(d,"Test Plot")