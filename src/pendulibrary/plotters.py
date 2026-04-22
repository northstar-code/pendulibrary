import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from os import listdir
from scipy.interpolate import CubicSpline


from pendulibrary.interpolate import interp_hermite
from pendulibrary.common_targetters import single_fixed
from pendulibrary.integrate import integrate_state
from pendulibrary.targeter import dc_underconstrained
from pendulibrary.common import hamiltonian

try:
    from plotly.express.colors import sample_colorscale
    from plotly.graph_objects import Figure, Scatter
except ImportError:
    print("Failed plotly import, some plotters may be unavailable")
try:
    from dash import Dash, dcc, html, Input, Output, State, Patch, callback
    import dash_bootstrap_components as dbc
    from dash.exceptions import PreventUpdate
except ImportError:
    print("Failed dash import, GUI will be unavailable")

def plot_timeline(
    xs_preprop: np.ndarray,
    ts_preprop: np.ndarray,
    fs_preprop: np.ndarray,
    Lr: float,
    N: int,
):
    Tf = ts_preprop[-1]
    _, xs_interp = interp_hermite(
        ts_preprop, xs_preprop, fs_preprop, np.linspace(0, Tf, N, False)
    )

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_preprop[:2]

    fig, axs = plt.subplots(1, N, figsize=(4 * N, 6))

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    for ii in range(N):
        ax = axs[ii]

        ax.plot(
            [0, x1_interp[ii], x2_interp[ii]],
            [0, y1_interp[ii], y2_interp[ii]],
            "o-",
            color="red",
        )
        ax.plot(0, 0, "ok", ms=15)

        ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
        ax.plot(x2_curve, y2_curve, "-k", lw=0.5)
        ax.axis("equal")
        ax.set(xticks=[], yticks=[])
    return fig


def plot_timeline_grid(
    xs_preprop: np.ndarray,
    ts_preprop: np.ndarray,
    fs_preprop: np.ndarray,
    Lr: float,
    nrow: int = 2,
    ncol: int = 10,
):
    Tf = ts_preprop[-1]
    N = nrow * ncol
    _, xs_interp = interp_hermite(
        ts_preprop, xs_preprop, fs_preprop, np.linspace(0, Tf, N, True)
    )
    _, xs_dense = interp_hermite(ts_preprop, xs_preprop, fs_preprop, n_mult=5)

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_dense[:2]

    fig, axs_grid = plt.subplots(nrow, ncol, figsize=(4 * ncol, 6 * nrow))
    axs = axs_grid.ravel()

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    for ii in range(N):
        ax = axs[ii]

        ax.plot(0, 0, "ok", ms=15)

        ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
        ax.plot(x2_curve, y2_curve, "-k", lw=0.5)

        ax.plot(
            [0, x1_interp[ii], x2_interp[ii]],
            [0, y1_interp[ii], y2_interp[ii]],
            "o-",
            color="red",
        )

        ax.axis("equal")
        ax.set(xticks=[], yticks=[])
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def make_gif(
    xs: np.ndarray,
    ts: np.ndarray,
    fs: np.ndarray,
    Lr: float,
    file_name: str,
    frames: int = 100,
    figsize: tuple = (5, 5),
    fps: int = 30,
):
    if not file_name.endswith(".gif"):
        file_name = file_name + ".gif"
    Tf = ts[-1]
    _, xs_interp = interp_hermite(ts, xs, fs, np.linspace(0, Tf, frames, False))
    _, xs_dense = interp_hermite(ts, xs, fs, n_mult=3)
    # xs_dense = xs.copy()

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_dense[:2]

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    fig, ax = plt.subplots(figsize=figsize)
    (line,) = ax.plot(
        [0, x1_interp[0], x2_interp[0]],
        [0, y1_interp[0], y2_interp[0]],
        "o-",
        color="red",
    )
    ax.plot(0, 0, "ok", ms=15)
    ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
    ax.plot(x2_curve, y2_curve, "-k", lw=0.5)
    ax.set(xticks=[], yticks=[])
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.axis("equal")

    def update(frame):
        line.set_xdata([0, x1_interp[frame], x2_interp[frame]])
        line.set_ydata([0, y1_interp[frame], y2_interp[frame]])
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    writer = animation.PillowWriter(fps=fps)
    anim.save(file_name, writer=writer)
    plt.close(fig)


def plot_timeline_grid(
    xs_preprop: np.ndarray,
    ts_preprop: np.ndarray,
    fs_preprop: np.ndarray,
    Lr: float,
    nrow: int = 2,
    ncol: int = 10,
):
    Tf = ts_preprop[-1]
    N = nrow * ncol
    _, xs_interp = interp_hermite(
        ts_preprop, xs_preprop, fs_preprop, np.linspace(0, Tf, N, True)
    )
    _, xs_dense = interp_hermite(ts_preprop, xs_preprop, fs_preprop, n_mult=5)

    t1, t2 = xs_interp[:2]
    t1_dense, t2_dense = xs_dense[:2]

    fig, axs_grid = plt.subplots(nrow, ncol, figsize=(4 * ncol, 6 * nrow))
    axs = axs_grid.ravel()

    y1_curve = -np.cos(t1_dense)
    x1_curve = np.sin(t1_dense)

    y2_curve = y1_curve - Lr * np.cos(t2_dense)
    x2_curve = x1_curve + Lr * np.sin(t2_dense)

    y1_interp = -np.cos(t1)
    x1_interp = np.sin(t1)
    y2_interp = y1_interp - Lr * np.cos(t2)
    x2_interp = x1_interp + Lr * np.sin(t2)

    for ii in range(N):
        ax = axs[ii]

        ax.plot(0, 0, "ok", ms=15)

        ax.plot(x1_curve, y1_curve, "-k", lw=0.5)
        ax.plot(x2_curve, y2_curve, "-k", lw=0.5)

        ax.plot(
            [0, x1_interp[ii], x2_interp[ii]],
            [0, y1_interp[ii], y2_interp[ii]],
            "o-",
            color="red",
        )

        ax.axis("equal")
        ax.set(xticks=[], yticks=[])
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def plot_nu_functions(
    eigs_all: np.ndarray, max_period_mult: int = 5, cmap: str = "hsv"
):
    fig, ax = plt.subplots()
    nus = np.sum(eigs_all, axis=1).real - 2
    colormap = plt.get_cmap(cmap)
    colors = [colormap(j / max_period_mult) for j in range(max_period_mult)]
    funcs = [nus - 2 * np.cos(2 * np.pi / (n + 1)) for n in range(max_period_mult)]
    funcs = np.array(funcs)
    xs = np.arange(len(eigs_all))
    ax.plot([0, xs[-1]], [0, 0], "--")

    for func, c, z in zip(funcs, colors, range(max_period_mult)):
        ax.plot(xs, func, "-", color=c, label=f"P{z+1}")

    ylim = 5
    try:
        inds_in = np.argwhere(np.any(np.abs(np.array(funcs)) < ylim, axis=0))
        indmax = np.max(inds_in) * 1.05

        ax.set(ylim=(-ylim, ylim), xlim=(0, indmax))
    except ValueError:
        pass
    ax.legend()
    return fig


def compare_fams(
    Xs_new: np.ndarray,
    fam_names: list,
    cmap: str = "hsv",
    figsize: tuple = (6, 6),
    ind_skip: int = 0,
):
    states = [r"$\theta_1$", r"$\theta_2$", r"$\omega_1$", r"$\omega_2$", r"T"]
    states.pop(ind_skip)
    fig, axs = plt.subplots(3, 3, figsize=figsize, sharex="col", sharey="row")

    for jj in range(3):
        for ii in range(jj):
            axs[-jj - 1, -ii - 1].remove()

    cmp = plt.get_cmap(cmap)
    N = len(fam_names)
    colrs = [cmp(j / N) for j in range(N)]
    for j in range(N):
        fname = fam_names[j]
        data = np.load(f"../database/{fname}.npz")
        x0s = data["x0s"]
        periods = data["periods"]
        Xs = np.column_stack((x0s, periods)).T
        Xs = np.delete(Xs, ind_skip, 0)
        # Xs = np.delete(Xs, 0, 0)
        for jj in range(3):
            for ii in range(jj + 1):
                axs[jj, ii].plot(Xs[ii + 1], Xs[jj], c=colrs[j])

        axs[0, 0].plot(np.nan, np.nan, c=colrs[j], label=fname)

    for jj in range(3):
        for ii in range(jj + 1):
            axs[jj, ii].plot(Xs_new[:, ii + 1], Xs_new[:, jj], ".k")
            axs[jj, ii].plot(Xs_new[-1, ii + 1], Xs_new[-1, jj], "xk")
            axs[jj, ii].plot(Xs_new[0, ii + 1], Xs_new[0, jj], "ok")
            # if ii != 0:
            #     axs[jj, ii].set(yticklabels=[])
            # if jj != 2:
            #     axs[jj, ii].set(xticklabels=[])

    for jj in range(3):
        axs[jj, 0].set(ylabel=states[jj])
        axs[-1, jj].set(xlabel=states[jj + 1])
    fig.legend()
    # axlg.set_axis_off()
    fig.tight_layout()
    return fig


def compare_fast(
    periods: np.ndarray,
    hamiltonians: np.ndarray,
    filenames: list,
    directory: str = "../database/",
    colormap:str='hsv'
):
    directory = directory.rstrip("/")
    vals_dict = {}
    for fname in filenames:
        data = np.load(f"{directory}/{fname}.npz")
        vals_dict[fname] = dict(T=data["periods"], H=data["hamiltonians"])

    per_range = max(periods) - min(periods)
    ham_range = max(hamiltonians) - min(hamiltonians)
    xlim = (max(periods) + min(periods)) / 2 + np.array([-per_range / 2, per_range / 2])
    ylim = (max(hamiltonians) + min(hamiltonians)) / 2 + np.array(
        [-ham_range / 2, ham_range / 2]
    )

    curve1 = Scatter(
        x=periods,
        y=hamiltonians,
        name="Compare",
        hoverinfo="name",
        mode="lines",
        line=dict(width=1.5, color="white"),
        showlegend=False,
    )

    n = len(vals_dict)
    colors = sample_colorscale(colormap, n) if n > 1 else ["rgb(255,0,0)"]
    curves = [curve1]
    j = 0
    for name, dct in vals_dict.items():
        curve = Scatter(
            x=dct["T"],
            y=dct["H"],
            name=name,
            hoverinfo="name",
            mode="lines",
            line=dict(width=1, color=colors[j]),
        )
        j += 1
        curves.append(curve)

    curveend = Scatter(
        x=periods,
        y=hamiltonians,
        name="Compare",
        hoverinfo="name",
        mode="markers",
        marker=dict(color="white", size=4),
    )

    curves.append(curveend)
    fig = Figure(data=curves)
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        xaxis_range=list(xlim),
        yaxis_range=list(ylim),
        margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(title="Period"),
        yaxis=dict(title="Hamiltonian"),
        modebar_remove=["autoScale", "lasso2d", "select2d", "toImage"],
        modebar_orientation='v'
    )

    config = dict(displaylogo=False, displayModeBar=True)
    fig.show(config=config)


def gui(
    db_path:str,
    targ_tol: float = 1e-8,
    int_tol: float = 1e-13,
    density: int = 5,
    port: int = 8050,
    n_time: int = 500,
    framerate=20,
    max_iter:int=15,
    debug:bool=False
):
    filenames = [file.removesuffix(".npz") for file in listdir(db_path) if file.endswith(".npz")]
    filenames = ["Select Family"] + filenames
    curve_inner = Scatter(
        x=[np.nan,np.nan],
        y=[np.nan,np.nan],
        mode="lines", hoverinfo=None, line=dict(color='darkgray', width=0.5)
    )
    curve_outer = Scatter(
        x=[np.nan,np.nan],
        y=[np.nan,np.nan],
        mode="lines", hoverinfo=None, line=dict(color='gray', width=1)
    )
    dots = Scatter(
        x=[0.,0.,0.],
        y=[0.,-1.,-2.],
        mode="markers+lines",hoverinfo=None,
        marker=dict(color='white', size=[15,10,10])
    )
    fig = Figure(data=[curve_inner, curve_outer, dots])

    # set layout
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=30, b=0, t=0),
        xaxis_visible=False, yaxis_visible=False
    )
    fig.update_yaxes(range=(-2.5,0.5),autorange=False)
    fig.update_xaxes(range=(-1.5,1.5),autorange=False)
    fig.update_yaxes(scaleanchor="x",scaleratio=1)

    # Dash dropdowns
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.themes.CYBORG])
    
    #fmt: off
    app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div([dbc.FormText("Family Name"),
                      dbc.Select(filenames, "Select Family", id="fam-dropdown")]),
            width=4
        ),
        dbc.Col(
            html.Div([dbc.FormText("Flip Family"),html.Br(),
                      dbc.Button("Flip", id="flip-h",size='sm')
                      ]),
            width=2
        ),
        dbc.Col(html.Div([dbc.FormText("Select Within Family"),
            dcc.Slider(0.0, 1.0, 1e-3, value=0, marks=None, included=False, updatemode="drag", id="slider")]),
            width=6
        )
            
    ]),

    dbc.Row([
        dbc.Col(html.Div([
                dbc.FormText("Animation Speed"),
                dcc.Slider(1, 50, 1, value=5, marks=None, included=False, id="speed")
            ]),
            width=4
        ),
        dbc.Col(
            html.Div([
                dbc.FormText("Animate"), html.Br(),
                dbc.Button("Play / Pause", id="play",style={"padding": "0px"}),
            ]),
            width=2
        ),
        
        dbc.Col(
            html.Div([
                dbc.Card([html.Div(id="card-body-content", children="Initial Condition",style={'lineHeight': '0.0', 'fontFamily': 'Consolas, "Courier New", monospace'})],body=True)
            ]),
            width=6
        ),
    ]),
    
    dbc.Row([dbc.Col(html.Div([dcc.Graph(figure=fig, id="display")]), width=12)]),

    html.Div([dcc.Store(id="curve-data", storage_type="memory"), 
        dcc.Store(id="k-store", storage_type="memory"), 
        dcc.Store(id="aux-data", storage_type="memory"),
        dcc.Interval(id="timer", interval=1000 / framerate, disabled=True)
    ])
    ], fluid=True)
    # fmt: on
    
    
    @callback(Output("aux-data", 'data',allow_duplicate=True), Output("slider", "value"), Input("fam-dropdown", "value"),prevent_initial_call=True)
    def set_family(fam):
        if fam == filenames[0]:
            raise PreventUpdate
        data = np.load(f"{db_path}/{fam}.npz")
        vals = np.column_stack((data['x0s'], data['periods']))
        
        # arclength (parameterize by this)
        arclen = np.append(0, np.cumsum(np.linalg.norm(np.diff(vals, axis=0), axis=1)))
        smax = max(arclen)
        Mr, Lr = data["params"]
        aux_data = {"arclen":arclen, "smax":smax,"vals":vals, "Mr":Mr,"Lr":Lr}
        
        return aux_data, 0.



    @callback(
        Output("display", "figure", allow_duplicate=True),
        Output("k-store", "data", allow_duplicate=True),
        Input("timer", "n_intervals"),
        State("k-store", "data"),
        State("speed", "value"),
        State("curve-data", "data"),
        State("timer", "disabled"),
        prevent_initial_call=True,
    )
    def animate(_, k,speed, curvedata, paused):
        if curvedata is None:
            raise PreventUpdate
        if paused:
            raise PreventUpdate

        curve = curvedata['coords']
        
        n = len(curve)
        k = k + speed
        k %= n

        patch = Patch()
        patch.data[2].x = [0.,curve[k][0],curve[k][2]]
        patch.data[2].y = [0.,curve[k][1],curve[k][3]]

        return patch, k
    
    @callback(
        Output("aux-data", "data", allow_duplicate=True),
        Input("flip-h", "n_clicks"),
        State("aux-data", "data"),
        prevent_initial_call=True,
    )
    def flip_horiz(_, aux_data):
        if aux_data is None:
            raise PreventUpdate
        
        aux_data['vals'] = np.array(aux_data['vals'])
        aux_data['vals'][:,:4] *= -1
        return aux_data

    @callback(
        Output("curve-data", "data",allow_duplicate=True),
        Output("k-store", "data"),
        Output("card-body-content", "children"),
        Output("display", "figure", allow_duplicate=True),
        Input("slider", "value"),
        Input("aux-data", "data"),
        State("display", "figure"),
        prevent_initial_call=True,
    )
    def update_curve_within_fam(s, aux_data, fig):
        if aux_data is None:
            raise PreventUpdate
        arclen, smax, vals = aux_data["arclen"], aux_data["smax"], aux_data["vals"]
        Lr,Mr = aux_data["Lr"], aux_data["Mr"]
        if vals is None:
            raise PreventUpdate
        spline = CubicSpline(arclen, vals, axis=0)
        
        patch = Patch()
        point = spline(s * smax)
        # point = np.array([np.interp(s*smax, arclen, v) for v in np.array(vals).T])    
        # print(point)
        x0, tf = point[:4], point[-1]
        targ = single_fixed(0, x0[0], 2, Lr, Mr, int_tol)
        Xg = targ.get_X(x0, tf)
        try:
            X, _, stm = dc_underconstrained(Xg, targ.g_dg_stm, targ_tol,max_iter=max_iter,debug=False)
        except RuntimeError:
            return aux_data, 0., [html.P(f"FAILED"), html.P(""), html.P("")],patch
        eigs = np.linalg.eigvals(stm)
        stab = np.max([(np.abs(lam) + 1 / np.abs(lam)) / 2 for lam in eigs])
        x0 = targ.get_x0(X)
        ham = hamiltonian(x0, Lr, Mr)
        tf = targ.get_period(X)
        ts, xs1, fs = integrate_state(x0, tf, Lr, Mr, int_tol)
        
        _, xs_t = interp_hermite(ts, xs1, fs, np.linspace(0, tf, n_time, True))
        _, xs_curve = interp_hermite(ts, xs1, fs, n_mult=density)

        y1_curve = -np.cos(xs_curve[0])
        x1_curve = np.sin(xs_curve[0])
        y2_curve = y1_curve - Lr * np.cos(xs_curve[1])
        x2_curve = x1_curve + Lr * np.sin(xs_curve[1])

        y1_t = -np.cos(xs_t[0])
        x1_t = np.sin(xs_t[0])
        y2_t = y1_t - Lr * np.cos(xs_t[1])
        x2_t = x1_t + Lr * np.sin(xs_t[1])
        
        maxx = max(np.max(np.abs(x1_t)),np.max(np.abs(x2_t)))
        maxy = max(0, y1_curve.max(),y2_curve.max())
        miny = min(0, y1_curve.min(),y2_curve.min())
        xl = np.array([-maxx,maxx])*1.1
        yl = np.array([miny, maxy])
        yl = np.mean(yl) + (yl-np.mean(yl))*1.1

        # lims = np.array([xl, yl])
        # ctrs = np.mean(lims, axis=1)
        # w = np.max(lims[:, 1] - lims[:, 0])
        # bounds = np.array([-w / 2, w / 2])

        
        # xl, yl = ctrs[:, None] + 1.3 * bounds[None, :]

        patch.layout.xaxis.range = xl
        patch.layout.yaxis.range = yl

        patch.data[0].x = x1_curve
        patch.data[0].y = y1_curve
        patch.data[1].x = x2_curve
        patch.data[1].y = y2_curve
        patch.data[2].x = [0.,x1_t[0],x2_t[0]]
        patch.data[2].y = [0.,y1_t[0],y2_t[0]]


        
        
        curve_data = dict(coords=np.array([x1_t,y1_t,x2_t,y2_t]).T,h=ham, stab=stab,period=tf)
        
        displaydata = [html.P(f"th0: [{x0[0]:.5f}, {x0[1]:.5f}]"), 
                       html.P(f"w0: [{x0[2]:.5f}, {x0[3]:.5f}]"), 
                       html.P(f"Period: {tf:.5f}, H: {ham:.4f}, SI: {stab:.2e}")]
        return curve_data, 0, displaydata, patch

    @callback(
        Output("timer", "disabled"),
        Input("play", "n_clicks"),
        State("timer", "disabled"),
        prevent_initial_call=True,
    )
    def pauseplay(_, disabled):
        return not disabled
    
    # print("COMPILING HELPERS...")
    data, _ = set_family("DDsp")        
    _ = update_curve_within_fam(0., data, fig) # this will force compile
    
    # print("\t\tCompiled")
    
    app.run(debug=debug, use_reloader=False, jupyter_mode="inline", port=port)

