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



# ##Reading in data file -- array of positions.
dat = np.genfromtxt(sys.argv[1])
# ##Ensure that the data is centered
dat = dat - np.median(dat, axis=0)
##Min and Max separation of interest -- currently passed as cmd line arguments
dmin = float(sys.argv[2])
dmax = float(sys.argv[3])
##Number of bins
nbins = 100
##Bins to use.
absc = np.linspace(dmin, dmax, nbins)
##Generating the control sample -- may need to think about size more
ncontrol = int((3. * dmax / absc[1])**1.5 * 10**.5 * 3)
ncontrol = ncontrol // dat.shape[1] * dat.shape[1]
print("Size of control sample", ncontrol)
control_pos = np.random.uniform(-1.5 * dmax, 1.5 * dmax, ncontrol)
control_pos.shape = (-1, dat.shape[1])
np.savez("two_point_func", absc, get_two_point(dat, control_pos, absc))

