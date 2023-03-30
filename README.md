# starforge_binary_search

Idenfies all of the binaries in a starforge simulation snapshot.

usage: find_multiples_new2.py [-h] [--sma_order]
                              [--halo_mass_file HALO_MASS_FILE]
                              [--mult_max MULT_MAX]
                              snap

Parse starforge snapshot, and get multiple data.

positional arguments:
  snap                  Name of snapshot to read

optional arguments:
  -h, --help            show this help message and exit
  --sma_order           Assemble hierarchy by sma instead of binding energy
  --halo_mass_file HALO_MASS_FILE
                        Name of file containing gas halo mass around sink
                        particles
  --mult_max MULT_MAX   Multiplicity cut (4). Max multiplicity systems to look for.
  
  Output: Two pickle files with all of the binary data stored. One pickle file (with TidesFalse) has the multiples without any cuts from tidal forces. 
  The other one includes the cut from tidal forces. 
 
  
  

