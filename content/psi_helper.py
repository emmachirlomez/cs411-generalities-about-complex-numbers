# try:
#     import micropip
#     micropip.install('ipywidgets')
#     micropip.install('ipympl')
# except:
#     pass

from ipywidgets import interact, interactive, fixed, interact_manual
import matplotlib.pyplot as plt
import numpy as np

def f(a, b):
    x = a + 1j * b
    return x ** 2 + 1

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

    print(f"Plot for the function f(X) = X^2 +8 *X + 12, in the point X = {real_part} + {imag_part} * i:")
    
    # old_figure = fig
    plt.show()
    return


