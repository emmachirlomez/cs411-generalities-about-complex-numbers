from ipywidgets import interact, interactive, fixed, interact_manual, widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def f(a, b):
    x = a + 1j * b
    return x ** 2 + 1

def real_solutions_function(a):
    return a ** 2 + 2 * a - 15

def more_difficult_example(a, b):
    x = a + 1j * b
    return (x + 4 - 2j)*(x + 4 + 2j)

# fig, ax = None, None
old_figure = None

def show_plots(real_part, imag_part):
    global old_figure
    if old_figure is not None:
        old_figure.close()
    
    # if fig is None:
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))
    # old_figure = fig
    
    # delete previous graph
    # for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    #     ax[i][j].cla()

    window_size = 5 # TODO: Change
    fig.tight_layout(pad=3.0)
    
    real_window = np.linspace(real_part - window_size, real_part + window_size, 200)
    imag_window = np.linspace(imag_part - window_size, imag_part + window_size, 200)

    real_f_from_real_changing = np.array([ f(i, imag_part).real for i in real_window])
    imag_f_from_real_changing = np.array([ f(i, imag_part).imag for i in real_window])
    real_f_from_imag_changing = np.array([ f(real_part, i).real for i in imag_window])
    imag_f_from_imag_changing = np.array([ f(real_part, i).imag for i in imag_window])
    
    abs_f_from_real_changing = np.array([ np.abs(f(i, imag_part)) for i in real_window])
    abs_f_from_imag_changing = np.array([ np.abs(f(real_part, i)) for i in imag_window])

    ax[0][0].plot(real_window, real_f_from_real_changing, label='f(X).real')
    ax[0][0].plot([real_part], [f(real_part, imag_part).real], 'r.', markersize=15)
    ax[0][0].plot(real_window, imag_f_from_real_changing, label='f(X).imag')
    ax[0][0].plot([real_part], [f(real_part, imag_part).imag], 'r.', markersize=15)
    ax[0][0].plot([real_part], [0.], 'g.', markersize=15)
    ax[0][0].set_xlabel("X.real")
    ax[0][0].set_title(f"Change in the real value of X, with X.imag={imag_part}")
    ax[0][0].set_xticks([i + real_part for i in range(-window_size, window_size + 1)])

    ax[0][1].plot(imag_window, real_f_from_imag_changing, label='f(X).real')
    ax[0][1].plot([imag_part], [f(real_part, imag_part).real], 'r.', markersize=15)
    ax[0][1].plot(imag_window, imag_f_from_imag_changing, label='f(X).imag')
    ax[0][1].plot([imag_part], [f(real_part, imag_part).imag], 'r.', markersize=15)
    ax[0][1].plot([imag_part], [0.], 'g.', markersize=15)
    ax[0][1].set_xlabel("X.imag")
    ax[0][1].set_title(f"Change in the imaginary value of X, with X.real={real_part}")
    ax[0][1].set_xticks([i + imag_part for i in range(-window_size, window_size + 1)])

    ax[1][0].plot(real_window, abs_f_from_real_changing, label='|f(X)|')
    ax[1][0].plot(real_window, 0 * imag_window, 'g', label='Target |f(X)| = 0')
    ax[1][0].plot([real_part], [np.abs(f(real_part, imag_part))], 'r.', markersize=15)
    ax[1][0].set_xlabel("X.real")
    ax[1][0].set_title(f"Change in the real value of X, with X.imag={imag_part}")
    ax[1][0].set_xticks([i + real_part for i in range(-window_size, window_size + 1)])

    ax[1][1].plot(imag_window, abs_f_from_imag_changing, label='|f(X)|')
    ax[1][1].plot(imag_window, 0 * imag_window, 'g', label='Target |f(X)| = 0')
    ax[1][1].plot([imag_part], [np.abs(f(real_part, imag_part))], 'r.', markersize=15)
    ax[1][1].set_xlabel("X.imag")
    ax[1][1].set_title(f"Change in the imaginary value of X, with X.real={real_part}")
    ax[1][1].set_xticks([i + imag_part for i in range(-window_size, window_size + 1)])
    
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax[i][j].grid(alpha=.9, which='both', linestyle='--')
        ax[i][j].legend()
        ax[i][j].set_ylim([-5, 5])

    plt.show()
    return

def show_plots_real_solution(x_value):
    global old_figure
    if old_figure is not None:
        old_figure.close()

    fig, ax = plt.subplots(figsize=(8, 5))

    window_size = 8 # TODO: Change
    fig.tight_layout(pad=3.0)
    
    real_window = np.linspace(x_value - window_size, x_value + window_size, 200)

    real_f_from_real_changing = np.array([ real_solutions_function(i) for i in real_window])

    real_function_zero_value = np.array([0 for i in range(len(real_window))])

    ax.plot(real_window, real_f_from_real_changing, label='f(X)')
    ax.plot([x_value], [real_solutions_function(x_value)], 'r.', markersize=15)
    ax.plot(real_window, real_function_zero_value, label='f(X) = 0')
    ax.plot([x_value], [0.], 'g.', markersize=15)
    ax.set_xlabel("X")
    ax.set_ylabel("f(X)")
    ax.set_title(f"Change in the real value of X")
    ax.set_xticks([i + x_value for i in range(-window_size, window_size + 1)])
  
    ax.grid(alpha=.9, which='both', linestyle='--')
    ax.legend()
    ax.set_ylim([-20, 20])
    
    plt.show()
    return

def show_more_difficult_example(real_part, imag_part):
    global old_figure
    if old_figure is not None:
        old_figure.close()
        
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 7))

    window_size = 5 # TODO: Change
    fig.tight_layout(pad=3.0)
    
    real_window = np.linspace(real_part - window_size, real_part + window_size, 200)
    imag_window = np.linspace(imag_part - window_size, imag_part + window_size, 200)

    real_f_from_real_changing = np.array([ more_difficult_example(i, imag_part).real for i in real_window])
    imag_f_from_real_changing = np.array([ more_difficult_example(i, imag_part).imag for i in real_window])
    real_f_from_imag_changing = np.array([ more_difficult_example(real_part, i).real for i in imag_window])
    imag_f_from_imag_changing = np.array([ more_difficult_example(real_part, i).imag for i in imag_window])
    
    abs_f_from_real_changing = np.array([ np.abs(more_difficult_example(i, imag_part)) for i in real_window])
    abs_f_from_imag_changing = np.array([ np.abs(more_difficult_example(real_part, i)) for i in imag_window])

    ax[0][0].plot(real_window, real_f_from_real_changing, label='f(X).real')
    ax[0][0].plot([real_part], [more_difficult_example(real_part, imag_part).real], 'r.', markersize=15)
    ax[0][0].plot(real_window, imag_f_from_real_changing, label='f(X).imag')
    ax[0][0].plot([real_part], [more_difficult_example(real_part, imag_part).imag], 'r.', markersize=15)
    ax[0][0].plot([real_part], [0.], 'g.', markersize=15)
    ax[0][0].set_xlabel("X.real")
    ax[0][0].set_title(f"Change in the real value of X, with X.imag={imag_part}")
    ax[0][0].set_xticks([i + real_part for i in range(-window_size, window_size + 1)])

    ax[0][1].plot(imag_window, real_f_from_imag_changing, label='f(X).real')
    ax[0][1].plot([imag_part], [more_difficult_example(real_part, imag_part).real], 'r.', markersize=15)
    ax[0][1].plot(imag_window, imag_f_from_imag_changing, label='f(X).imag')
    ax[0][1].plot([imag_part], [more_difficult_example(real_part, imag_part).imag], 'r.', markersize=15)
    ax[0][1].plot([imag_part], [0.], 'g.', markersize=15)
    ax[0][1].set_xlabel("X.imag")
    ax[0][1].set_title(f"Change in the imaginary value of X, with X.real={real_part}")
    ax[0][1].set_xticks([i + imag_part for i in range(-window_size, window_size + 1)])

    ax[1][0].plot(real_window, abs_f_from_real_changing, label='|f(X)|')
    ax[1][0].plot(real_window, 0 * imag_window, 'g', label='Target |f(X)| = 0')
    ax[1][0].plot([real_part], [np.abs(more_difficult_example(real_part, imag_part))], 'r.', markersize=15)
    ax[1][0].set_xlabel("X.real")
    ax[1][0].set_title(f"Change in the real value of X, with X.imag={imag_part}")
    ax[1][0].set_xticks([i + real_part for i in range(-window_size, window_size + 1)])

    ax[1][1].plot(imag_window, abs_f_from_imag_changing, label='|f(X)|')
    ax[1][1].plot(imag_window, 0 * imag_window, 'g', label='Target |f(X)| = 0')
    ax[1][1].plot([imag_part], [np.abs(more_difficult_example(real_part, imag_part))], 'r.', markersize=15)
    ax[1][1].set_xlabel("X.imag")
    ax[1][1].set_title(f"Change in the imaginary value of X, with X.real={real_part}")
    ax[1][1].set_xticks([i + imag_part for i in range(-window_size, window_size + 1)])
    
    for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax[i][j].grid(alpha=.9, which='both', linestyle='--')
        ax[i][j].legend()
        #ax[i][j].set_ylim([-5, 5])

    plt.show()
    return

def exercise_1():
    x = widgets.IntSlider(min=-10, max=10, step=1, value=0)
    interact(show_plots_real_solution, x_value=x);

def exercise_2():
    real_part = widgets.IntSlider(min=-10, max=10, step=1, value=0)
    imag_part = widgets.IntSlider(min=-10, max=10, step=1, value=0)

    interact(show_plots, real_part=real_part, imag_part=imag_part);

def exercise_3():
    real_part = widgets.IntSlider(min=-10, max=10, step=1, value=0)
    imag_part = widgets.IntSlider(min=-10, max=10, step=1, value=0)

    interact(show_more_difficult_example, real_part=real_part, imag_part=imag_part);

def fun_fact():
    plot_4d_function(lambda a: a**2+1, -2, 2, -2, 2)

def get_coordonates_of_function(f, real_min=-10, real_max=10, img_min=-10, img_max=10):
    range_x = np.arange(real_min, real_max, 0.1)
    range_y = np.arange(img_min, img_max, 0.1)
    Xp, Yp = np.meshgrid(range_x, range_y)
    nr_lin, nr_col = Xp.shape
    result = np.array(
        [[f(x + y * 1j) for x in range_x]for y in range_y]
    )
    Zp = np.real(result)
    Tp = np.imag(result)
    return Xp, Yp, Zp, Tp

def plot_4d_function(f, real_min=-10, real_max=10, img_min=-10, img_max=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot a basic wireframe.
    X, Y, Z, T = get_coordonates_of_function(f, real_min=real_min, real_max=real_max, img_min=img_min, img_max=img_max)
    C = T - np.amin(T)
    C = C / max(0.1, np.amax(C))
    
    surf = ax.plot_surface(X, Y, Z, facecolors=cm.jet(C), alpha =0.8)

    ax.set_xlabel('a.real axis')
    ax.set_ylabel('a.imag axis')
    ax.set_zlabel('f(a).real axis')
    fig.colorbar(surf)
    
    plt.show();

    
    