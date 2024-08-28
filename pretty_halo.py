import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
import seaborn as sns
import pickle
import h5py

from meshoid import Meshoid

from labelLine import labelLines
import cgs_const as cgs
sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
# from analyze_multiples import snap_lookup
import find_multiples_new2
from find_multiples_new2 import cluster, system
import starforge_constants as sfc
import configparser

colorblind_palette = sns.color_palette("colorblind")
# Set the matplotlib color cycle to the seaborn colorblind palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['patch.linewidth'] = 3

def get_phalo(base, aa, snap_idx, bin_id1, bin_id2, my_ft):
    with open(base + aa + "/path_lookup.p", "rb") as ff:
        path_lookup = (pickle.load(ff))

    tmp_pos = path_lookup[f"{bin_id1}"][snap_idx, 2:5]
    with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{snap_idx}_compFalse_tf{my_ft}.hdf5") as hf:
        tmp_halo_pos = (hf[f"halo_{bin_id1}_x"][...])
        tmp_halo_mass = (hf[f"halo_{bin_id1}_m"][...])
        tmp_halo_rho = (hf[f"halo_{bin_id1}_rho"][...])

    tmp_pos2 = path_lookup[f"{bin_id2}"][snap_idx, 2:5]
    with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{snap_idx}_compFalse_tf{my_ft}.hdf5") as hf:
        tmp_halo_pos2 = (hf[f"halo_{bin_id2}_x"][...])
        tmp_halo_mass2 = (hf[f"halo_{bin_id2}_m"][...])
        tmp_halo_rho2 = (hf[f"halo_{bin_id2}_rho"][...])

    center = np.copy(tmp_pos)
    tmp_pos_center = tmp_pos - center
    tmp_halo_pos_center = tmp_halo_pos - center
    tmp_pos2_center = tmp_pos2 - center
    tmp_halo_pos2_center = tmp_halo_pos2 - center

    return center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center

def get_phalo_limits(base, aa, snap_idx, bin_id1, bin_id2):
    center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center = get_phalo(base, aa, snap_idx,
                                                                                                   bin_id1, bin_id2, "8.0")
    ##Automatically set axis extent based on the size of the halos -- TO DO PLOT CONSI
    halos_x = (np.concatenate(
        (tmp_halo_pos_center[:, 0], tmp_halo_pos2_center[:, 0], [tmp_pos_center[0]], [tmp_pos2_center[0]])))
    halos_y = (np.concatenate(
        (tmp_halo_pos_center[:, 1], tmp_halo_pos2_center[:, 1], [tmp_pos_center[1]], [tmp_pos2_center[1]])))
    xmin, xmax = min(halos_x), max(halos_x)
    ymin, ymax = min(halos_y), max(halos_y)

    return xmin, xmax, ymin, ymax

config = configparser.ConfigParser()
config.read("config")

snap_idx = config.getint("params","snap_idx")
bin_id1 = config.getint("params","bin1")
bin_id2 = config.getint("params", "bin2")
my_ft = config.get("params","ft", fallback="8.0")
seed = config.getint("params","seed", fallback=42)
rmax = config.getfloat("params", "rmax", fallback=0.5)
res = config.getint("params", "res", fallback=800)
savetype = config.get("params","savetype", fallback="pdf")
d_cut = rmax

base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}/"
r2 = f"_TidesTrue_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"

snap_file = base + f"snapshot_{snap_idx:03d}.hdf5"
den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base = \
find_multiples_new2.load_data(snap_file, res_limit=1e-3)
##WOULD HAVE BEEN BETTER TO PUT THIS FUNCTIONALITY IN LOAD DATA!!!
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

##GET HALO AND PARTICLE POSITIONS CENTERED ON THE FIRST STAR
center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center = get_phalo(base, aa, snap_idx,
                                                                                       bin_id1, bin_id2, my_ft)
##ONLY SELECT GAS IN VOXEL AROUND STARS
sel2 = np.abs(xuniq - center)
sel2 = (sel2[:,0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:,2] < d_cut)

##GETTING SURFACE DENSITY VIA THE MESHOID PACKAGE
xuniq_center = xuniq - center
M = Meshoid(xuniq_center[sel2], muniq[sel2], huniq[sel2])
X = np.linspace(- rmax, rmax, res)
Y = np.linspace(- rmax, rmax, res)
X, Y = np.meshgrid(X, Y, indexing='ij')
sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,  size=2 * rmax, res=res, center=np.array((0,0,0)))  # *1e4

############################################################################################################

fig,ax = plt.subplots(constrained_layout=True)
ax.set_xlabel("x [pc]")
ax.set_ylabel("y [pc]")
p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(), cmap="viridis", linewidth=0, rasterized=True)
sc1 = ax.scatter(tmp_pos_center[0], tmp_pos_center[1], marker="o", alpha=0.3, edgecolors="k", facecolors="none")
ax.scatter(tmp_pos2_center[0], tmp_pos2_center[1],  marker="o", alpha=0.3, edgecolors="k", facecolors="none")
plt.colorbar(p, label=r"$\Sigma$ [$M_{\odot} pc^{-2}$]")

fig.savefig("pretty." + savetype)

############################################################################################################

fig,ax = plt.subplots(constrained_layout=True)
ax.set_xlabel("x [pc]")
ax.set_ylabel("y [pc]")

xmin, xmax, ymin, ymax = get_phalo_limits(base, aa, snap_idx, bin_id1, bin_id2)
buff = 0.35
ax.set_xlim(xmin -  buff * (xmax - xmin), xmax + buff * (xmax - xmin))
ax.set_ylim(ymin -  buff * (ymax - ymin), ymax + buff * (ymax - ymin))

p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(), cmap="viridis", linewidth=0, rasterized=True)
ax.scatter(tmp_halo_pos_center[:, 0], tmp_halo_pos_center[:, 1], color=colorblind_palette[0], alpha=0.15)
sc1 = ax.scatter(tmp_pos_center[0], tmp_pos_center[1], c="k", marker="X")
ax.scatter(tmp_halo_pos2_center[:, 0], tmp_halo_pos2_center[:, 1], color=colorblind_palette[1], alpha=0.15)
ax.scatter(tmp_pos2_center[0], tmp_pos2_center[1], c="k", marker="X")

plt.colorbar(p, label=r"$\Sigma$ [$M_{\odot} pc^{-2}$]")
fig.savefig("prettyb." + savetype)