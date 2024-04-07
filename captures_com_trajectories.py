import numpy as np
from matplotlib import pyplot as plt
import sys
import pickle

sys.path.append("/home/aleksey/code/python/")
sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
from analyze_multiples import snap_lookup
import find_multiples_new2
import cgs_const as cgs



LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7

def subtract_path(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff

def path_divide(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones(len(p1)) * np.inf
    filt = (~np.isinf(p1)) & (~np.isinf(p2))
    diff[filt] = p1[filt] / p2[filt]

    return diff

##READING IN AUXILIARY DATA TABLES...############################################################################################
sim_tag = "M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42"
base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/"
r2 = sys.argv[1].replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"
base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
start_snap = 100
end_snap = 489
lookup = np.load(base + aa + "/system_lookup_table.npz")['arr_0']
bin_ids = np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
ic = np.load(base + aa + "/ic.npz", allow_pickle=True)['arr_0']
final_bins_arr_id = np.load(base + aa + "/final_bins_arr_id.npz")["arr_0"]
bin_class = np.load(base + aa + "/classes.npz", allow_pickle=True)['arr_0']

t_first = ic[:, 0]
conv = cgs.pc / cgs.au
snap_interval = 2.47e4
##TRANSFORMATION TO PARTICLE ...############################################################################################
sinks_all = []
ts = []
tags = []
nsinks = np.zeros(end_snap + 1)
for ss in range(start_snap, end_snap + 1):
    tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
    sinks_all.append(tmp_sink)
    ts.append(ss * np.ones(len(tmp_sink)))

sinks_all = np.vstack(sinks_all)
ts = np.concatenate(ts)
ts.shape = (-1, 1)
sinks_all = np.hstack((ts, sinks_all))
sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))

tags = ["{0}_{1}".format(sinks_all[ii, 1], sinks_all[ii, 0]) for ii in range(len(ts))]
sinks_all = sinks_all[np.argsort(tags)]
tags2 = ["{0}_{1}".format(lookup[ii, 1], lookup[ii, 0]) for ii in range(len(ts))]
lookup_sorted = lookup[np.argsort(tags2)]
sinks_all = np.hstack((sinks_all, lookup_sorted[:,[2, LOOKUP_MTOT, LOOKUP_SMA, LOOKUP_ECC]]))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
assert(np.all(np.array(tags)[np.argsort(tags)] == np.array(tags2)[np.argsort(tags2)]))
######Saving a path for each particle
utags = np.unique(sinks_all[:, 1])
utags_str = utags.astype(int).astype(str)
path_lookup = {}
for ii, uu in enumerate(utags):
    tmp_sel = sinks_all[sinks_all[:, 1] == uu]
    tmp_path1 = np.ones((end_snap + 1, len(sink_cols))) * np.inf
    tmp_path1[tmp_sel[:, 0].astype(int)] = tmp_sel

    path_lookup[utags_str[ii]] = tmp_path1

with open(base + aa + "/path_lookup.p", "wb") as ff:
    pickle.dump(path_lookup, ff)

mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]
##MAKING PLOTS OF TRAJECTORIES FOR ALL OF THE BINARIES ...############################################################################################
fig_idx = 0
dens_series = np.zeros((len(bin_ids), end_snap + 1))
halo_snap = np.zeros(len(bin_ids))

for jj in range(len(bin_ids)):
    if not np.isin(jj, final_bins_arr_id):
        continue

    tmp_row = np.array(list(bin_ids[jj])).astype(str)
    tmp_class = bin_class[jj]
    ##Getting com over time for pair
    tmp1 = path_lookup[tmp_row[0]]
    ms1 = tmp1[:, mcol]
    ms1[np.isinf(ms1)] = 1
    ms1.shape = (-1, 1)
    tmp2 = path_lookup[tmp_row[1]]
    ms2 = tmp2[:, mcol]
    ms2[np.isinf(ms2)] = 1
    ms2.shape = (-1, 1)
    coms = (tmp1[:, pxcol:pzcol + 1] * ms1 + tmp2[:, pxcol:pzcol + 1] * ms2) / (ms1 + ms2)
    ##Positions in the evolving com frame
    p1 = subtract_path(tmp1[:, pxcol:pzcol + 1], coms)  # [100:]
    p2 = subtract_path(tmp2[:, pxcol:pzcol + 1], coms)  # [100:]
    psep = subtract_path(tmp1[:, pxcol:pzcol + 1], tmp2[:, pxcol:pzcol + 1])
    psep = np.sum((psep * psep)**.5, axis=1)
    tmp_filt = ~np.isinf(p1[:, 0])

    print(tmp_row[0])
    sys1_info = lookup[int(tmp_row[0]) == lookup[:, LOOKUP_PID].astype(int)]
    sys2_info = lookup[int(tmp_row[1]) == lookup[:, LOOKUP_PID].astype(int)]

    ##Getting initial semimajor axis -- need to do this ringamarole in case the
    ##stars start unbound from each other...
    tmp1_fst = tmp1[tmp_filt][0]
    tmp2_fst = tmp2[tmp_filt][0]
    tmp_orb = find_multiples_new2.get_orbit(tmp1_fst[pxcol:pzcol+1], tmp2_fst[pxcol:pzcol+1],\
                                            tmp1_fst[vxcol:vzcol+1], tmp2_fst[vxcol:vzcol+1],\
                                            tmp1_fst[mtotcol], tmp2_fst[mtotcol],\
                                           tmp1_fst[hcol], tmp2_fst[hcol])


    # coms = np.ones((end_snap + 1, 3)) * np.inf
    ##Get orbits(!!!)
    ##Just have to do this once at the FST(!)

    ##Have to take care of nans here...
    closest_approaches = np.zeros(len(utags))
    closest_approaches_norm = np.zeros(len(utags))
    closest_approaches_time = np.zeros(len(utags))
    closest_approaches_norm_time = np.zeros(len(utags))

    path_diff_all = []
    mass_all = []
    for ii, uu in enumerate(utags_str):
        #Want only closest approach of stars external to the binary.
        if uu in tmp_row:
            continue
        ##Displacement from binary com
        path_diff = subtract_path(path_lookup[uu][:, pxcol:pzcol + 1], coms)
        path_diff = np.sum(path_diff * path_diff, axis=1)**.5
        path_diff_all.append(path_diff)
        mass_all.append(path_lookup[uu][:, mcol])
        ##Closest approach for this particle
        order = np.argsort(path_diff)
        closest_approaches_time[ii] = path_lookup[uu][order[0]][0] * snap_interval
        closest_approaches[ii] = path_diff[order[0]] * conv
        ##Get separation normalized by binary separation
        path_diff_norm = path_divide(path_diff, psep)
        order_norm = np.argsort(path_diff_norm)
        ##Time where (normalized) closest approach occurs
        closest_approaches_norm_time[ii] = path_lookup[uu][order_norm[0]][0] * snap_interval
        ##Normalized closest approach
        closest_approaches_norm[ii] = path_diff_norm[order_norm[0]]
        # closest_approaches_norm[ii] = path_diff[order[0]]
        # print(path_lookup[uu][:, 0])
    ##Transpose distance and mass array so that the rows are times.
    path_diff_all = np.array(path_diff_all).T
    mass_all = np.array(mass_all).T
    ##Ordering particles by distance at every snapshot
    path_diff_all_order = np.argsort(path_diff_all, axis=1)
    path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
    mass_all = np.take_along_axis(mass_all, path_diff_all_order, axis=1)
    ##Compute time series of the local density for this binary, using the 32 closes particles...
    dens_series[jj] = path_divide(np.sum(mass_all[:, :32], axis=1), path_diff_all[:, 31] ** 3.)

    ##Minimum closest approach over all particles
    order = np.argsort(closest_approaches)
    closest_tags = utags_str[order]
    closest_approaches = closest_approaches[order]
    closest_approaches_time = closest_approaches_time[order]
    #Minimum normalized closest approach over all particles
    order_norm = np.argsort(closest_approaches_norm)
    closest_tags_norm = utags_str[order_norm]
    closest_approaches_norm = closest_approaches_norm[order_norm]
    closest_approaches_norm_time = closest_approaches_norm_time[order_norm]

    ##Halo masses
    tmp_halo_mass = sys1_info[:, LOOKUP_MTOT] - sys1_info[:, LOOKUP_M]
    halo_snap1_list = np.where(tmp_halo_mass > 0.01 * sys1_info[:, LOOKUP_M])[0]
    tmp_halo_mass = sys2_info[:, LOOKUP_MTOT] - sys2_info[:, LOOKUP_M]
    halo_snap2_list = np.where(tmp_halo_mass > 0.01 * sys2_info[:, LOOKUP_M])[0]

    ##Last snapshot with significant gas(!)
    halo_snap1 = 0
    halo_snap2 = 0
    if len(halo_snap1_list) > 0:
        halo_snap1 = halo_snap1[-1]
    if len(halo_snap2_list) > 0:
        halo_snap2 = halo_snap2[-1]
    halo_snap[jj] = max(halo_snap1, halo_snap2)
    # fig, axs = plt.subplots(figsize=(20, 10), ncols=2)
    # fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    # delta = np.abs(p1[tmp_filt][0] - p2[tmp_filt][0]) * conv
    #
    # # ax = axs[0]
    # ax.set_xlabel("x [au]")
    # ax.set_ylabel("y [au]")
    # ax.set_xlim(-1.1 * delta[0], 1.1 * delta[0])
    # ax.set_ylim(-1.1 * delta[1], 1.1 * delta[1])
    #
    # filt_together = (tmp1[:, scol] == tmp2[:, scol]) & (tmp_filt)
    # segs = np.where(~filt_together[:-1] == filt_together[1:])[0]
    # segs = np.append(segs, len(filt_together) - 1)
    #
    # ax.plot(p1[:, 0] * conv, p1[:, 1] * conv, 'r--', alpha=0.3)
    # ax.plot(p2[:, 0] * conv, p2[:, 1] * conv, 'b--', alpha=0.3)
    # seg_last = -1
    # for tmp_seg in segs:
    #     start = seg_last + 1
    #     end = tmp_seg + 1
    #     if filt_together[start]:
    #         ax.plot(p1[:, 0][start:end] * conv, p1[:, 1][start:end] * conv, 'r-', alpha=0.6)
    #         ax.plot(p2[:, 0][start:end] * conv, p2[:, 1][start:end] * conv, 'b-', alpha=0.6)
    #
    #     seg_last = tmp_seg
    # ax.annotate(r"$a_i = {0:.0f}$ au, $e_i$ = {1:.2g}".format(tmp_orb[0] * cgs.pc / cgs.au, tmp_orb[1]) + "\n" + "{0}".format(
    #         tmp_row), (0.01, 0.99), ha='left', va='top', xycoords='axes fraction')
    #
    # if fig_idx == 15:
    #     tag = closest_tags[2]
    #     tmp_path = path_lookup[tag]
    #     pclose = subtract_path(tmp_path[:, pxcol:pzcol + 1], coms)
    #     for ii in range(len(p1)):
    #         fig, axs = plt.subplots(figsize=(20, 10), ncols=2)
    #         ax = axs[0]
    #         ax.set_xlabel("x [au]")
    #         ax.set_ylabel("y [au]")
    #         ax.set_xlim(-3.1 * delta[0], 3.1 * delta[0])
    #         ax.set_ylim(-3.1 * delta[1], 3.1 * delta[1])
    #
    #         ax.plot(p1[ii, 0] * conv, p1[ii, 1] * conv, 'rs')
    #         ax.plot(p2[ii, 0] * conv, p2[ii, 1] * conv, 'bs')
    #         ax.plot(pclose[ii, 0] * conv, pclose[ii, 1] * conv, 'o', color='0.5')
    #
    #         ax = axs[1]
    #         ax.set_ylim(0,1)
    #         ax.set_xlabel("t [yr]")
    #         ax.set_ylabel("e")
    #         ax.plot(sys1_info[:, LOOKUP_SNAP] * snap_interval, sys1_info[:, LOOKUP_ECC], color='purple')
    #         idx = np.where(sys1_info[:, LOOKUP_SNAP] == ii)[0]
    #         if len(idx) > 0:
    #             ax.plot(sys1_info[idx, LOOKUP_SNAP] * snap_interval, sys1_info[idx, LOOKUP_ECC], 'ks', color='purple')
    #
    #         fig.savefig(base + aa + "break_com_path_{0:03d}_{1}.png".format(fig_idx, ii))
    #
    #         plt.clf()

    # ax = axs[1]
    # ax.set_xlabel("x [pc]")
    # ax.set_ylabel("y [pc]")
    #
    # ax.plot(tmp1[:, pxcol], tmp1[:, pycol], 'r', alpha=0.5)
    # ax.plot(tmp2[:, pxcol], tmp2[:, pycol], 'b--', alpha=0.5)
    # ax.plot(tmp1_fst[pxcol], tmp2_fst[pycol], "rs")
    #
    # for tag in closest_tags[2:3]:
    #     tmp_path = path_lookup[tag]
    #     ax.plot(tmp_path[:, pxcol], tmp_path[:, pycol], color='0.5')
    #
    # for tag in closest_tags_norm[2:3]:
    #     tmp_path = path_lookup[tag]
    #     ax.plot(tmp_path[:, pxcol], tmp_path[:, pycol], '--', color='0.5')
    #
    # close_str1 = r"Closest distance ={0:.2g} au, time = {1:.2g} yr".format(closest_approaches[2], closest_approaches_time[2]) +\
    #             "\n" + "{0}\n".format(closest_tags[2])
    # close_str2 = r"Min (Approach / Bin Sep) ={0:.2g}, time = {1:.2g} yr".format(closest_approaches_norm[2], closest_approaches_norm_time[2]) +\
    #             "\n" + "{0}\n".format(closest_tags_norm[2])
    # close_str3 = "Class: {0}".format(tmp_class)
    # ax.annotate(close_str1 + close_str2 + close_str3, (0.01, 0.99), ha='left', va='top', xycoords='axes fraction', fontsize=16)
    # ##Could also add the time of the closest encounter...
    # fig.savefig(base + aa + "com_path_{0:03d}.pdf".format(fig_idx))
    # fig.savefig(base + aa + "tmp_com_path_{0:03d}.png".format(fig_idx))
    # fig_idx += 1
    # plt.clf()
np.savez("dens_series_final_bins.npz", dens_series, halo_snap)
###########################################################################