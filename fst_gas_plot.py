import pickle
import numpy as np
import find_multiples_new2
import argparse
import subprocess

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from meshoid import Meshoid


def bash_command(cmd, **kwargs):
    '''Run command from the bash shell'''
    process = subprocess.Popen(['/bin/bash', '-c', cmd], **kwargs)
    return process.communicate()[0]


def main():
    ##Fix GN from the simulation data rather than hard-coding...
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Index of snapshot to read")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")
    parser.add_argument("--ptag", default="", help="Extra tag for auxiliary files")

    args = parser.parse_args()

    snap_idx = args.snap

    snap_file = args.snap_base + '_{0}.hdf5'.format(snap_idx)
    den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base = \
        find_multiples_new2.load_data(snap_file, res_limit=1e-3)

    xuniq, indx = np.unique(x, return_index=True, axis=0)
    muniq = m[indx]
    huniq = h[indx]
    vuniq = v[indx]
    uuniq = u[indx]
    denuniq = den[indx]
    vuniq = vuniq.astype(np.float64)
    xuniq = xuniq.astype(np.float64)
    muniq = muniq.astype(np.float64)
    huniq = huniq.astype(np.float64)
    uuniq = uuniq.astype(np.float64)
    denuniq = denuniq.astype(np.float64)
    partpos = partpos.astype(np.float64)
    partmasses = partmasses.astype(np.float64)
    partsink = partsink.astype(np.float64)
    center = np.median(xuniq, axis=0)
    xuniq -= center

    rmax = 20
    radius_cut = np.sum(xuniq * xuniq, axis=1) < (rmax * rmax)
    xuniq = xuniq[radius_cut]
    muniq = muniq[radius_cut]
    huniq = huniq[radius_cut]

    first_snap_idx = np.genfromtxt("first_snap_idx" + args.ptag)
    first_together = np.genfromtxt("first_together" + args.ptag)
    bin_ids = np.load("unique_bin_ids.npz", allow_pickle=True)['arr_0']
    # lookup_ref = np.load("system_lookup_table.npz")['arr_0']

    M = Meshoid(xuniq, muniq, huniq)
    res = 800
    X = np.linspace(-rmax, rmax, res)
    Y = np.linspace(-rmax, rmax, res)
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(figsize=(8, 8))
    sigma_gas_msun_pc2 = M.SurfaceDensity(M.m, center=np.zeros(3), size=2. * rmax, res=res)  # *1e4
    p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=.1, vmax=1e3))
    ax.set_aspect('equal')
    fig.colorbar(p, label=r"$\Sigma_{gas}$ $(\rm M_\odot\,pc^{-2})$")
    ax.set_xlabel("X (pc)")
    ax.set_ylabel("Y (pc)")

    filt_new_stars = np.in1d(partids, first_snap_idx[first_snap_idx[:, -1] <= int(snap_idx)][:, 0])
    if len((partpos[:, 0] - center[0])[filt_new_stars]) > 0:
        ax.scatter((partpos[:, 0] - center[0])[filt_new_stars], (partpos[:, 1] - center[1])[filt_new_stars], color='0.5', s=3)

    bin_sel = bin_ids[(first_together <= int(snap_idx))]
    ##Only plot binary stars that have not been dissociated yet??
    for row in bin_sel:
        row_li = list(row)

        pos1 = partpos[np.where(partids == row_li[0])[0]]
        pos2 = partpos[np.where(partids == row_li[1])[0]]
        ax.plot(pos1[0] - center[0], pos1[1] - center[1])
        ax.plot(pos2[0] - center[0], pos2[1] - center[1])


    fig.savefig("gas_plot_{0}.pdf".format(snap_idx))
    fig.savefig("gas_plot_{0}.png".format(snap_idx))



if __name__ == "__main__":
    main()
