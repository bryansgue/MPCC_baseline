import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
plt.rc('text', usetex = True)
def fancy_plots_2():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    return fig, ax1, ax2


def fancy_plot():
    # Define parameters fancy plot
    pts_per_inch = 72.27
    # write "\the\textwidth" (or "\showthe\columnwidth" for a 2 collumn text)
    text_width_in_pts = 300.0
    # inside a figure environment in latex, the result will be on the
    # dvi/pdf next to the figure. See url above.
    text_width_in_inches = text_width_in_pts / pts_per_inch
    # make rectangles with a nice proportion
    golden_ratio = 0.618
    # figure.png or figure.eps will be intentionally larger, because it is prettier
    inverse_latex_scale = 2
    # when compiling latex code, use
    # \includegraphics[scale=(1/inverse_latex_scale)]{figure}
    # we want the figure to occupy 2/3 (for example) of the text width
    fig_proportion = (3.0 / 3.0)
    csize = inverse_latex_scale * fig_proportion * text_width_in_inches
    # always 1.0 on the first argument
    fig_size = (1.0 * csize, 0.7 * csize)
    # find out the fontsize of your latex text, and put it here
    text_size = inverse_latex_scale * 10
    label_size = inverse_latex_scale * 10
    tick_size = inverse_latex_scale * 8

    params = {'backend': 'ps',
            'axes.labelsize': text_size,
            'legend.fontsize': tick_size,
            'legend.handlelength': 2.5,
            'legend.borderaxespad': 0,
            'xtick.labelsize': tick_size,
            'ytick.labelsize': tick_size,
            'font.family': 'serif',
            'font.size': text_size,
            # Times, Palatino, New Century Schoolbook,
            # Bookman, Computer Modern Roman
            # 'font.serif': ['Times'],
            'ps.usedistiller': 'xpdf',
            'text.usetex': True,
            'figure.figsize': fig_size,
            # include here any neede package for latex
            'text.latex.preamble': [r'\usepackage{amsmath}',
                ],
                }
    plt.rc(params)
    plt.clf()
    # figsize accepts only inches.
    fig = plt.figure(1, figsize=fig_size)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.13,
                        hspace=0.05, wspace=0.02)
    plt.ioff()
    ax1 = fig.add_subplot(111)
    return fig, ax1

def plot_pose(x, xref, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$']
    
    for i in range(3):
        ax.plot(t[0:x.shape[1]], x[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])
        
        ax.plot(t[0:x.shape[1]], xref[i, 0:x.shape[1]],
                color=colors[i], lw=2, ls="--", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_control(u, t):
    fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Colores para cada control
    labels = [r'$F$', r'$Tx$', r'$Ty$', r'$Tz$']  # Etiquetas para cada control
    
    for i in range(4):
        axs[i].plot(t[0:u.shape[1]], u[i, :], color=colors[i], lw=2, ls="-")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    axs[-1].set_xlabel(r"$[t]$", labelpad=5)
    
    fig.tight_layout()
    
    return fig

def plot_error(error, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(3):
        ax.plot(t[0:error.shape[1]], error[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_vel_lineal(v, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB']  # Add color for psi
    labels = [r'$x_p$', r'$y_p$', r'$z_p$']
    
    for i in range(3):
        ax.plot(t[0:v.shape[1]], v[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[vel_lineal]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_vel_angular(w, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51', '#5189BB']  # Add color for psi
    labels = [r'$\omega_x$', r'$\omega_y$', r'$\omega_z$']
    
    for i in range(3):
        ax.plot(t[0:w.shape[1]], w[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[vel_angular]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_CBF(value, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651']  # Add color for psi
    labels = [r'$value$']
    
    for i in range(1):
        ax.plot(t[0:value.shape[1]], value[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])

    ax.set_ylabel(r"$[CBF_value]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_vel_norm(value, value_ref, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651', '#69BB51']  # Add color for psi
    labels = [r'$x$', r'$y$']
    
    for i in range(1):
        ax.plot(t[0:value.shape[1]], value[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])
        ax.plot(t[0:value.shape[1]], value_ref[i, 0:value.shape[1]],
                color=colors[i], lw=2, ls="--", label=labels[i] + r'$d$')
    

    ax.set_ylabel(r"$[CBF_value]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_distance(value, t):
    fig, ax = fancy_plot()
    
    colors = ['#BB5651']  # Add color for psi
    labels = [r'$value$']
    
    for i in range(1):
        ax.plot(t[0:value.shape[1]], value[i, :],
                color=colors[i], lw=2, ls="-", label=labels[i])
    
    # Dibujar la línea horizontal en el valor 0.9
    ax.axhline(y=0.9, color='blue', linestyle='--', linewidth=1.5, label='Threshold = 0.9')
    
    ax.set_ylabel(r"$[Distance]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig

def plot_time(ts, delta_t, t):
    fig, ax = fancy_plot()
    ax.set_xlim((t[0], t[-1]))
    
    colors = ['#BB5651', '#69BB51', '#5189BB', '#FFD700']  # Add color for psi
    labels = [r'$x$', r'$y$', r'$z$', r'$\psi$']
    
    for i in range(1):
        ax.plot(t[0:ts.shape[1]], ts[i, :],
                color='#BB5651', lw=2, ls="--", label=labels[i])
        
        ax.plot(t[0:ts.shape[1]], delta_t[i, 0:ts.shape[1]],
                color='#69BB51', lw=2, ls="-", label=labels[i] + r'$d$')

    ax.set_ylabel(r"$[states]$", rotation='vertical')
    ax.set_xlabel(r"$[t]$", labelpad=5)
    
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=False, ncol=2,
              borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
              borderaxespad=0.3, columnspacing=2)
    
    ax.grid(color='#949494', linestyle='-.', linewidth=0.5)
    
    return fig


def plot_progress_velocity(v_theta, v_real, theta, t):
    """Plot v_θ (solver input) vs v_real (drone projection) and θ progress.

    Parameters
    ----------
    v_theta : ndarray (1, N)  – progress velocity from solver
    v_real  : ndarray (1, N)  – real progress speed (tangent · v)
    theta   : ndarray (1, N+1) – arc-length state θ
    t       : ndarray          – time vector
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    N = v_theta.shape[1]
    t_plot = t[0:N]

    # ── Top: v_θ vs v_real ───────────────────────────────────────────────
    ax1.plot(t_plot, v_theta[0, :], color='#BB5651', lw=2, ls="-",
             label=r'$v_{\theta}$ (solver input)')
    ax1.plot(t_plot, v_real[0, :], color='#5189BB', lw=2, ls="-",
             label=r'$v_{real} = \mathbf{t}^T \mathbf{v}$')
    ax1.set_ylabel(r"Velocity $[\mathrm{m/s}]$")
    ax1.legend(loc="best", frameon=True, fancybox=True, shadow=False)
    ax1.grid(color='#949494', linestyle='-.', linewidth=0.5)
    ax1.set_title(r'Progress velocity: $v_{\theta}$ vs real drone speed along path')

    # ── Middle: ratio v_real / v_θ ───────────────────────────────────────
    ratio = v_real[0, :] / (v_theta[0, :] + 1e-8)
    ax2.plot(t_plot, ratio, color='#69BB51', lw=2, ls="-",
             label=r'$v_{real} / v_{\theta}$')
    ax2.axhline(y=1.0, color='#333333', linestyle='--', linewidth=1.0,
                label='Ideal ratio = 1')
    ax2.set_ylabel(r"Ratio")
    ax2.set_ylim([-0.5, 2.5])
    ax2.legend(loc="best", frameon=True, fancybox=True, shadow=False)
    ax2.grid(color='#949494', linestyle='-.', linewidth=0.5)

    # ── Bottom: θ progress ───────────────────────────────────────────────
    # theta has shape (1, N+1); plot only first N points aligned with t_plot
    theta_plot = theta[0, :N]
    ax3.plot(t_plot, theta_plot, color='#BB5651', lw=2, ls="-",
             label=r'$\theta(t)$')
    ax3.set_ylabel(r"$\theta$ $[\mathrm{m}]$")
    ax3.set_xlabel(r"Time $[\mathrm{s}]$")
    ax3.legend(loc="best", frameon=True, fancybox=True, shadow=False)
    ax3.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig.tight_layout()
    return fig
