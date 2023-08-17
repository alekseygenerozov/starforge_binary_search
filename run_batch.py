import sys
import subprocess

def bash_command(cmd, **kwargs):
        '''Run command from the bash shell'''
        process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
        return process.communicate()[0]


# snap_base = "/scratch3/03532/mgrudic/STARFORGE_RT/STARFORGE_v1.1/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/output/snapshot"
start = int(sys.argv[1]) 
interval = 20
if len(sys.argv)==3:
        interval = int(sys.argv[2])
end = start + interval + 1
print(start, end)
##FOR TESTING
# start = 250
# end = 251
# snap_base = "snapshot"
for ii in range(start, end, 1):
        bash_command("python3 find_multiples_new2.py --halo_mass_file M2e4halo_masses_sing_\
npTrue_c0.5 --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
        bash_command("python3 find_multiples_new2.py --halo_mass_file M2e4halo_masses_sing_\
npTrue_c0.5 --compress --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
        bash_command("python3 find_multiples_new2.py --halo_mass_file M2e4halo_masses_sing_\
npTrue_c0.5 --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 1.0))

        bash_command("python3 find_multiples_new2.py --ngrid 4 --snap_base {0} {1} --tides_factor {2} ".format(snap_base, ii, 8.0))
        bash_command("python3 find_multiples_new2.py --compress --ngrid 4 --snap_base {0} {1} --tides_factor {2}".format(snap_base, ii, 8.0))
        bash_command("python3 find_multiples_new2.py --ngrid 4 --snap_base {0} {1} --tides_factor {2} ".format(snap_base, ii, 1.0))

