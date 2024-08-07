import numpy as np
import pickle
import sys
import h5py

import pytreegrav

sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
import starforge_constants as sfc
import find_multiples_new2

def get_mxvh(h5f, id):
    mgas1 = ff['halo_{0}_m'.format(id)][...]
    xgas1 = ff['halo_{0}_x'.format(id)][...]
    vgas1 = ff['halo_{0}_v'.format(id)][...]
    hgas1 = ff['halo_{0}_h'.format(id)][...]
    if mgas1.shape == (1, 2):
        mgas1 = np.zeros((0))
        xgas1 = np.zeros((0, 3))
        vgas1 = np.zeros((0, 3))
        hgas1 = np.zeros((0))
    #hgas1 = np.zeros(len(mgas1))

    return mgas1, xgas1, vgas1, hgas1

def KE(xc, mc, vc, uc):
    """ xc - array of positions
        mc - array of masses
        vc - array of velocities
        uc - array of internal energies
    """
    ## velocity w.r.t. com velocity
    v_bulk = np.average(vc, weights=mc, axis=0)
    v_well = vc - v_bulk
    vSqr = np.sum(v_well ** 2, axis=1)
    return (mc * (vSqr / 2 + uc)).sum()

def get_energies(p1_dat, g1_dat, p2_dat, g2_dat):
    pos1, vel1, m1, h1 = p1_dat
    pos2, vel2, m2, h2 = p2_dat
    mgas1, xgas1, vgas1, hgas1 = g1_dat
    mgas2, xgas2, vgas2, hgas2 = g2_dat

    blob_pos = np.vstack((np.atleast_2d(pos1), xgas1))
    blob_vel = np.vstack((np.atleast_2d(vel1), vgas1))
    blob_mass = np.concatenate(([m1], mgas1))
    blob_com = np.average(blob_pos, weights=blob_mass, axis=0)
    blob_com_vel = np.average(blob_vel, weights=blob_mass, axis=0)

    pe = m2 * pytreegrav.PotentialTarget(np.atleast_2d(pos2), blob_pos,
                                         blob_mass,
                                         softening_target=np.atleast_1d(h2),
                                         softening_source=np.concatenate(([h1], hgas1)),
                                         G=sfc.GN, method='bruteforce')[-1]

    ke = KE(np.vstack((blob_com, pos2)), np.append(np.sum(blob_mass), m2),
            np.vstack((blob_com_vel, vel2)), np.append(0, 0))

    return pe, ke


##Column lookup
##################################################################################################
sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vycol = np.where(sink_cols == "vy")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
mcol = np.where(sink_cols == "m")[0][0]
##Loading post-processed data tables
#########################################################################################################
sim_tag = f"M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{sys.argv[1]}"
base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{sys.argv[1]}/"
r2 = sys.argv[2].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
# start_snap = 100
# end_snap = 489
bin_ids = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
with open(base + aa + "/path_lookup.p", "rb") as ff:
    path_lookup = pickle.load(ff)
#########################################################################################################
en_col = []
print(len(bin_ids))
for ii, uid in enumerate(bin_ids):
    bin_list = list(uid)
    path1 = path_lookup[f"{bin_list[0]}"]
    path2 = path_lookup[f"{bin_list[1]}"]
    fst_idx = np.where(~np.isinf(path1[:,0]) & ~np.isinf(path2[:,0]))[0][0]
    print(fst_idx)

    pos1 = path1[fst_idx, pxcol:pzcol + 1]
    pos2 = path2[fst_idx, pxcol:pzcol + 1]
    vel1 = path1[fst_idx, vxcol:vzcol + 1]
    vel2 = path2[fst_idx, vxcol:vzcol + 1]
    m1 = path1[fst_idx, mcol]
    m2 = path2[fst_idx, mcol]
    h1 = path1[fst_idx, hcol]
    h2 = path2[fst_idx, hcol]
    ###!!!! PATH SHOULD ALIGN WITH COMMAND LINE ARGS.
    with h5py.File(base + f"/halo_masses/\
halo_masses_sing_npTrue_c0.5_{fst_idx}_compFalse_tf{sys.argv[3]}.0.hdf5", "r") as ff:
        g1_dat = get_mxvh(ff, bin_list[0])
        g2_dat = get_mxvh(ff, bin_list[0])

    ##How often does the inequality flip without the gas???
    if m1 + np.sum(g1_dat[0]) > m2 + np.sum(g2_dat[0]):
        tmp = get_energies((pos1, vel1, m1, h1), g1_dat, (pos2, vel2, m2, h2), g2_dat)
    else:
        tmp = get_energies((pos2, vel2, m2, h2), g2_dat, (pos1, vel1, m1, h1), g1_dat)

    en_col.append(tmp)

print(len(en_col))
np.savez(f"en_col_seed{sys.argv[1]}_{sys.argv[3]}.npz", en_col)