import sys
import subprocess

def bash_command(cmd, **kwargs):
	'''Run command from the bash shell'''
	process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
	return process.communicate()[0]


snap_base = "/scratch3/03532/mgrudic/STARFORGE_RT/STARFORGE_v1.1/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_42/output/snapshot"
##FOR TESTING
# snap_base = "snapshot"
start = int(sys.argv[1]) #* 10
end = (int(sys.argv[2]) + 1) #* 10
#end = start + 1
print(start, end)
for ii in range(start, end, 1):
	bash_command("python halo_masses_single_double_par.py --non_pair --tides_factor 8.0 --snap_base {0}  {1}".format(snap_base, ii))
	bash_command("python halo_masses_single_double_par.py --non_pair --tides_factor 8.0 --compress --snap_base {0}  {1}".format(snap_base, ii))
	bash_command("python halo_masses_single_double_par.py --non_pair --tides_factor 1.0 --snap_base {0}  {1}".format(snap_base, ii))
