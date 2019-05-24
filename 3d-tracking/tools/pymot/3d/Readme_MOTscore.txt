******************************************************
MOTscore.pl
Multiple Object Tracking Scoring Tool
Developer: Keni Bernardin, University of Karlsruhe, ISL
Date: 1.2007
******************************************************


To run the scoring tool, type at the command line
./MOTscore.pl GroundTruth_A.txt Hypos_A.txt yes/no

The script expects 3 parameters:
- The name of the 3D ground truth file (CHIL CLEAR format)
- The name of the tracker output file (CHIL CLEAR format)
- A flag, if the A-MOTA should be computed or not ('yes' or 'no')

The meaning of the last flag is the following:
It was agreed that for the acoustic source localization task, in this run, the system's capability to recognize when a speaker has changed will not be evaluated. Therefore, all ID mismatches will be ignored. The resulting A-MOTA should be computed for the audio person tracking task.
To calculate the A-MOTA, set the flag to 'yes'
To calculate just the MOTA, set the flag to 'no' (or anything else)


The output of the script are the computed error metrics according to the CLEAR Person Tracking Task Description
The final line summarizes:
MOTP(in mm)   Miss-rate   FalsePos-rate   Mismatch-rate   (Failed correspondences)? MOTA/A-MOTA(?)   (in %)

Reports on Bugs and other problems are welcome!
