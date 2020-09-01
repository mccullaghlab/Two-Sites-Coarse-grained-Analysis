import numpy as np
import MDAnalysis as md
from MDAnalysis.analysis import align
import reach_routines as reach
import henm 

selection1 = "backbone"
selection2 = "protein and not backbone and not resname GLY"

prmtop = "../TRAJ/traj.prmtop"
traj   = "../TRAJ/traj.dcd"

# setup MDAnalysis universe and selection
coord = md.Universe(prmtop,traj)
print("Number of frames:",coord.trajectory.n_frames)
sel1 = coord.select_atoms(selection1)
sel2 = coord.select_atoms(selection2)

avgPos, alignedPos  = reach.iterative_align_average_2_site(coord,sel1,sel2)
np.savetxt("igps_100ns_com_average_structure.dat",avgPos)
nSites = len(sel1.residues) + len(sel2.residues)

covar = np.zeros((nSites*3,nSites*3),dtype=np.float64)
Site_pos = np.zeros((nSites,3),dtype=float)
Site_pos = avgPos

out = open("igps_100ns_com_average_structure.pdb",'w')
out.write("TITLE     MDANALYSIS FRAME 0: Created by PDBWriter\n")
out.write("CRYST1   88.814   88.814   88.814  90.00  90.00  90.00 P 1           1\n")
count = 0
for res1 in range(len(sel1.residues)):
     out.write("ATOM  %5i  CA  %3s S%4i    %8.3f%8.3f%8.3f  1.00  0.00\n" %(count+1,sel1.residues[res1].resname,sel1.residues[res1].resid,avgPos[count][0],avgPos[count][1],avgPos[count][2]))
     count += 1
     for res2 in range(len(sel2.residues)):
          if sel1.residues[res1].resid == sel2.residues[res2].resid:
               out.write("ATOM  %5i  CB  %3s S%4i    %8.3f%8.3f%8.3f  1.00  0.00\n" %(count+1,sel2.residues[res2].resname,sel2.residues[res2].resid,avgPos[count][0],avgPos[count][1],avgPos[count][2]))
               count += 1
out.write("END\n")
out.close

# loop through trajectory and compute covariance 
sel = coord.select_atoms("bynum 1:865")
with md.Writer("igps_100ns_com_positions.dcd", nSites) as W:
     for ts in coord.trajectory:
          Site_pos = alignedPos[ts.frame-1,:,:]
          covar += np.dot(alignedPos[ts.frame-1,:,:].reshape(3*nSites,1),alignedPos[ts.frame-1,:,:].reshape(1,3*nSites))
          sel.atoms.positions = alignedPos[ts.frame]
          W.write(sel.atoms)

# finish covariance
covar /= coord.trajectory.n_frames
covar -= np.dot(avgPos.reshape(3*nSites,1),avgPos.reshape(1,3*nSites))
np.savetxt("igps_100ns_com_covar.dat",covar)
