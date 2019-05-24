******************************************************
scoreAll.py
Wrapper for Multiple Object Tracking Scoring Tool
Developer: Keni Bernardin, University of Karlsruhe, ISL
Date: 1.2007
******************************************************


To score submissions for a system and task, consisting of multiple output (.PT) files for multiple seminars and segments:
- Make sure to have all the submitted output .PT files for this task in one HYPO directory (e.g. UKA_3DMPT_V_EVAL06_PRIMARY/)
- Make sure no other .PT files are contained in this directory (also not in subdirectories!)
- Make sure all the naming conventions are respected in what concerns hypothesis files
An example HYPO directory could be:

  UKA_3DMPT_V_EVAL06_PRIMARY:
	AIT_20051010_SEGMENT1.PT
	AIT_20051011_B_SEGMENT1.PT
	AIT_20051011_C_SEGMENT1.PT
	AIT_20051011_D_SEGMENT1.PT
	IBM_20050819_SEGMENT1.PT
	IBM_20050822_SEGMENT1.PT
	IBM_20050823_SEGMENT1.PT
	IBM_20050830_SEGMENT1.PT
	UKA_3DMPT_V_EVAL06_PRIMARY.txt
	UPC_20050706_SEGMENT1.PT
	UPC_20050720_SEGMENT1.PT
	UPC_20050722_SEGMENT1.PT
	UPC_20050727_SEGMENT1.PT

- Make sure all the 3D labels are available in one Label directory tree(!). The structure of the directory tree must be exactly the same as in the official distribution. The expected LABEL directory tree is as follows:

  DEVEL_LABELS:
	AIT_20060728/
	IBM_20060720/
	ITC_20060714/
	UKA_20060726/
	UPC_20060613/

with
  AIT_20060728:
	AIT_20060728_3d_label.txt
	AIT_20060728_Acoustic3d_label.txt
	AIT_20060728_Multimodal3d_label.txt

  IBM_20060720:
	IBM_20060720_3d_label.txt
	IBM_20060720_Acoustic3d_label.txt
	IBM_20060720_Multimodal3d_label.txt
  etc...


- Be sure to use the appropriate flags and labels for the task you wish to score:
  V: 3d_labels			 for visual person tracking
  A: Acoustic3d_labels		 for acoustic person tracking
  M: Multimodal3d_labels	 for multimodal person tracking


To run the scoring tool, type at the command line
./scoreAll.py LABELdir HYPOdir FLAG

The script expects 3 parameters:
- The name of the LABEL directory
- The name of the HYPO directory for this task and system
- a flag, for selecting the right scoring mode. The right 3d label file names (e.g. "AIT_20060728_Acoustic3d_label.txt" or "IBM_20060720_3d_label.txt", etc... are automatically parsed using the hypo file name and the flag)

The scoring tool will output the MOTA for the visual and the multimodal person tracking subtasks. It will output the A-MOTA for the acoustic person tracking subtask.


The output of the script are the computed error metrics according to the CLEAR Person Tracking Task Description
MOTP(in mm)
Miss-rate(in %)
FalsePos-rate(in %)
Mismatches(absolute and in %)
MOTA(in %)

For the acoustic tracking subtask, the output is:
MOTP(in mm)
Miss-rate(in %)
FalsePos-rate(in %)
Share of the Miss/FalsePos rates caused by localization errors above the threshold of 500mm (in %)
A-MOTA(in %)

Reports on Bugs and other problems are welcome!
