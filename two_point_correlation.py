from scipy.spatial import cKDTree
import numpy as np
import sys
import matplotlib.pyplot as plt

def get_binned_dist_tree(p1, p2, absc):
    """
    Binned distance count

    :param p1: First set of positions
    :param p2: Second set of positions
    :param absc: Bins of radial distance

    :return: Counts of pairs within each distance bin, normalized by the total number of pairs
    """
    kd_tree1 = cKDTree(p1)
    kd_tree2 = cKDTree(p2)
    binned_dist = kd_tree1.count_neighbors(kd_tree2, absc, cumulative=False)
    ##Remove points less than the minimum distance
    binned_dist = binned_dist[1:]
    ##Number of pairs within specified range of distances...
    npairs = np.sum(binned_dist)

    return binned_dist / npairs


def get_two_point(d_pos, control_pos, absc):
    """
    Landy-Szalay estimator for the two-point correlation

    :param d_pos: Data set
    :param p2: Control positions
    :param absc: Bins of radial distance

    :return: Landy-Szalay estimator of the two-point correlation function
    """
    get_binned_dist = get_binned_dist_tree
    dist_bin_dd = get_binned_dist(d_pos, d_pos, absc)
    dist_bin_dc = get_binned_dist(d_pos, control_pos, absc)
    dist_bin_cc = get_binned_dist(control_pos, control_pos, absc)

    two_point = (dist_bin_dd - 2. * dist_bin_dc + dist_bin_cc) / dist_bin_cc

    return two_point

ncontrol = 3000
control_pos = np.random.uniform(0, 400, ncontrol)
control_pos.shape = (-1, 3)
control_pos_og = np.copy(control_pos)
thetas = np.arccos(np.random.uniform(-1, 1, len(control_pos_og)))
phis = np.random.uniform(0, 2. * np.pi, len(control_pos_og))
# control_pos_end = control_pos_og + 3.5 * np.transpose((np.cos(phis), np.sin(phis)))
rs = np.random.normal(loc = 10.5, scale=0.5, size=len(control_pos_og))
control_pos_end = control_pos_og +  np.transpose((rs * np.cos(phis) * np.sin(thetas), rs * np.sin(phis) * np.sin(thetas), rs * np.cos(thetas)))
# control_pos_end = control_pos_og +  np.transpose((rs * np.cos(phis), rs * np.sin(phis)))

dat = np.vstack((control_pos, control_pos_end))
# np.savetxt("twod_test", control_pos)


# ##Reading in data file -- array of positions.
# dat = np.genfromtxt(sys.argv[1])
# ##Ensure that the data is centered
dat = dat - np.median(dat, axis=0)
##Min and Max separation -- currently passed as cmd line arguments
dmin = float(sys.argv[2])
dmax = float(sys.argv[3])
##Number of bins
nbins = 100
##Bins to use.
absc = np.linspace(dmin, dmax, nbins)

fig, ax = plt.subplots()
ax.set_xscale('log')
fs = [16.0, 32.0]
col = 0
ls = ['-', '--']
for fidx,ff in enumerate(fs):
    ##Control sample -- also choose uniformly
    ncontrol2 =  int(ff * ncontrol)
    control_pos = np.random.uniform(-1.5 * dmax, 1.5 * dmax, ncontrol2)
    control_pos.shape = (-1, dat.shape[1])

    # np.savez("two_point_func", absc, get_two_point(dat, control_pos, absc))
    absc_cen = 0.5 * (absc[:-1] + absc[1:])
    plt.plot(absc_cen, get_two_point(dat, control_pos, absc), color=str(col), label=str(ff), linestyle=ls[fidx])
    col+=0.5
ax.legend()
plt.show()