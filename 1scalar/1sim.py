'''
Kuramoto model:

dq[i] / dt = ws[i] + sum[K / N * np.cos(q[j] - q[i]), j != i]
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', lw=3.5)
plt.rc('xtick', direction='in', top=True, bottom=True)
plt.rc('ytick', direction='in', left=True, right=True)

from scipy.integrate import solve_ivp

def get_dydt_scalar(ws, k):
    n = len(ws)
    ones = np.ones(n)

    def dydt(t, y):
        qis = np.outer(y, ones)
        diffs = qis.T - qis # diffs[i, j] = qis[j] - qis[i]
        return ws + k / n * np.sum(np.sin(diffs), axis=1)
    return dydt

# run a kuramoto simulation w/ n oscillators: n random phases + n random
# frequencies, chosen from distribution N(w0, sigm)
def run_kuramoto(n, w0, sigm, k, tf=100, tol=1e-8, **kwargs):
    y0 = np.random.rand(n) * 2 * np.pi
    ws = np.random.normal(w0, sigm, n) # can be negative
    dydt = get_dydt_scalar(ws, k)
    ret = solve_ivp(dydt, (0, tf), y0,
                    atol=tol, rtol=tol, dense_output=True, **kwargs)
    t_vals = np.linspace(ret.t.min(), ret.t.max(), 1000)
    q_vals = ret.sol(t_vals)
    qsin_mean = np.mean(np.sin(q_vals), axis=0)
    qcos_mean = np.mean(np.cos(q_vals), axis=0)
    mod = np.sqrt(qsin_mean**2 + qcos_mean**2)

    return t_vals, mod, ret

def analyze_kuramoto(t_vals, mod):
    ''' return rise time (reach 0.8 * max(mod)), asymptotic (mean of
    rise_time until end) '''
    maxmod = np.max(mod)
    # mean growth time to all of these = "rise time"
    threshes = np.linspace(0.6, 0.9, 5)
    rises = []
    for thresh in threshes:
        rises = np.where(mod > thresh * maxmod)[0][0]
    rise_idx = int(np.mean(rises))
    asymptote = np.mean(mod[rise_idx: ])
    return t_vals[rise_idx], asymptote, maxmod

def run_kscan(k_arr=np.geomspace(0.03, 0.3, 10), n=3000, sigm=0.03,
              w0=1, fn='1kval_scan', nruns=5, **kwargs):
    k_res = np.zeros((len(k_arr), nruns, 3))
    for k_idx, k_val in enumerate(k_arr):
        for run_idx in range(nruns):
            print('Running for', k_val, run_idx)
            t_vals, mod, _ = run_kuramoto(n, w0, sigm, k_val)
            k_res[k_idx, run_idx] = analyze_kuramoto(t_vals, mod)
    averaged_res = np.mean(k_res, axis=1)
    rise_times, asymptotes, maxes = np.array(averaged_res).T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                   gridspec_kw={'height_ratios': [1, 2] },
                                   sharex=True)
    ax1.plot(k_arr, rise_times)
    ax2.plot(k_arr, asymptotes, label='Asymptote')
    ax2.plot(k_arr, maxes, label='Maxes')
    ax2.legend(fontsize=14)
    ax2.set_xscale('log')

    ax2.set_xlabel(r'$K$')
    ax2.set_ylabel(r'$|r|$')
    ax1.set_ylabel(r'$\omega_0 t_{\rm rise}$')
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    plt.close()


if __name__ == '__main__':
    # n, w0, sigm, k = 100, 1, 0.1, 0.1
    # t_vals, mod, _ = run_kuramoto(n, w0, sigm, k)
    # plt.plot(t_vals, mod)
    # plt.savefig('/tmp/foo', dpi=200)

    run_kscan(n=300, fn='1kval_scan300')
    run_kscan()
    run_kscan(n=10000, fn='1kval_scan1e4')
