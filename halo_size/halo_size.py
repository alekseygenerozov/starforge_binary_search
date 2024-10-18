import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle
import sys

sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
import starforge_constants as sfc

import hydra

delta = (-.38, 0.22, -0.068, -0.42, 0.65)
a = (5.95, 6,18, 10.26, 7.71, 98.87)
b = (9.25, 9.89, 10.24, 11.13, 14.28)

def sigmoid(x):
    return 0.5 * (1. + x / (1. + x**2.)**.5)

def ad_index(u):
    u_cgs = u * 100**2.
    gamma = 5. / 3.
    for kk in range(5):
        gamma += delta[kk] * sigmoid(a[kk] * (np.log10(u_cgs) - b[kk]))

    return gamma

def u_to_cs(u1):
    gamma_eff = ad_index(u1)
    return u1**.5 * (gamma_eff * (gamma_eff - 1))**.5

def get_shape_eigen(dxc):
    """ dxc - distance from center (density max) position
    """
    ## Return length of principle axes
    evals, evecs = np.linalg.eig(np.cov(dxc.T)) # This seems very slow ...
    ord1 = np.argsort(evals)

    return evals[ord1]**.5, evecs[ord1]

def jeans(rho, cs):
    return cs / (sfc.GN  * rho)**.5

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config) ->None:
    for seed in (2, 42):
        my_ft = config.ft.ft
        print(my_ft)
        sim_tag = f"M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}"
        base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}/"
        r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/{0}/M2e4_snapshot_".format(sim_tag)
        r2 = f"_TidesTrue_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(".p", "")
        aa = "analyze_multiples_output_" + r2 + "/"
        base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
        bin_ids =  np.load(base + aa + "/unique_bin_ids.npz", allow_pickle=True)['arr_0']
        fst = np.load(base + aa + "/fst.npz", allow_pickle=True)['arr_0']
        with open(base + aa + "/path_lookup.p", "rb") as ff:
            path_lookup = pickle.load(ff)

        rhalos = []
        rjeans = []

        for ii, uid in enumerate(bin_ids):
            ss = fst[ii]
            with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{ss}_compFalse_tf{my_ft}.hdf5","r") as ff:
                    bin_row = list(bin_ids[ii])
                    sink_pos = path_lookup[str(bin_row[0])][ss, 2:5]

                    kk0 = f'halo_{bin_row[0]}'
                    kk = f'halo_{bin_row[0]}_x'
                    kk_rho = f'halo_{bin_row[0]}_rho'
                    kk_u = f'halo_{bin_row[0]}_u'

                    tmp_bound = ff[kk0][...]
                    tmp_pos = ff[kk][...]
                    tmp_rho = ff[kk_rho][...]
                    tmp_u = ff[kk_u][...]

                    if len(tmp_pos.shape) != 2:
                        print("bad", tmp_bound)
                    elif np.sum(tmp_pos) == 0:
                        print("bad", tmp_bound)
                    elif len(tmp_pos) < 10:
                        print("bad", tmp_bound)
                    else:
                        #dx = tmp_pos
                        # dx = tmp_pos - np.median(tmp_pos, axis=0)
                        rho_mean = np.mean(tmp_rho)
                        cs_mean = u_to_cs(np.mean(tmp_u))
                        dx = tmp_pos - sink_pos
                        print("cs:", cs_mean * 100 / 1e5)

                        evals, evecs = get_shape_eigen(dx)
                        a1, a2, a3 = evals
                        rhalos.append((a1 * a2 * a3)**(1./3.))
                        rjeans.append(jeans(rho_mean, cs_mean))

                        fig, axs = plt.subplots(figsize=(16, 8), ncols = 2, constrained_layout=True)
                        ax = axs[0]
                        ax.set_xlabel("x [pc]")
                        ax.set_ylabel("y [pc]")
                        ax.set_title(f"{a1:.3g} {a2:.3g} {a3:.3g} {rhalos[-1]:.3g}")
                        ax.scatter(dx[:,0], dx[:,1])
                        ax.plot([0, a1 * evecs[0,0]], [0, a1 * evecs[0,1]], "r--", linewidth=4)
                        ax.plot([0, a2 * evecs[1,0]], [0, a2 * evecs[1,1]], "r--", linewidth=4)
                        ax.plot([0, a3 * evecs[2,0]], [0, a3 * evecs[2,1]], "r--", linewidth=4)

                        ax = axs[1]
                        ax.set_xlabel("x [pc]")
                        ax.set_ylabel("z [pc]")
                        ax.scatter(dx[:,0], dx[:,2], c='k')
                        ax.plot([0, a1 * evecs[0,0]], [0, a1 * evecs[0,2]], "r--", linewidth=4)
                        ax.plot([0, a2 * evecs[1,0]], [0, a2 * evecs[1,2]], "r--", linewidth=4)
                        ax.plot([0, a3 * evecs[2,0]], [0, a3 * evecs[2,2]], "r--", linewidth=4)

                        fig.savefig(f"tmp_{seed}_{my_ft}_{ii:03d}.png")
                        plt.close()

        np.savez("halo_sizes.npz", rhalos=rhalos, rjeans=rjeans)

if __name__=="__main__":
    main()