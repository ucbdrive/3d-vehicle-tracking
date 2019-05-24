#! /usr/bin/python

import os
import sys


### helper function

def getCommandOutput(command):
    """run a command and return its stdout output"""
    child = os.popen(command)
    data = child.read()
    err = child.close()
    if err:
        print('%s failed w/ exit code %d' % (command, err))
        return None
    return data


### main

def main():
    # default 3d label file extension
    lab3dNameV = "3d_label.txt"
    lab3dNameA = "Acoustic3d_label.txt"
    lab3dNameM = "Multimodal3d_label.txt"

    argc = len(sys.argv)
    if argc == 4:
        for argind, arg in enumerate(sys.argv):
            if argind == 1:
                labelBase = arg
            if argind == 2:
                hypoPath = arg
            if argind == 3:
                mode = arg

    else:
        print(
            "Usage: scoreAll.py <label file base directory> <hypo file "
            "directory> <'V'/'A'/'M': "
            "Visual/Acoustic/Multimodal task>")
        sys.exit()

    if mode == "V":
        lab3dName = lab3dNameV
    elif mode == "A":
        lab3dName = lab3dNameA
    elif mode == "M":
        lab3dName = lab3dNameM
    else:
        print(
            "Only flags 'V', 'A' or 'M' for Visual, Acoustic or Multimodal "
            "task scoring are available")
        sys.exit()

    ### accumulation buffers

    groundT = 0.0
    corr = 0.0
    fcorr = 0.0
    distance = 0.0
    miss = 0.0
    falseP = 0.0
    mismatch = 0.0

    ### run scoring script

    for root, dirs, files in os.walk(hypoPath):
        files.sort()
    for file in files:
        fileEnding = file[-3:]
        if fileEnding == ".PT":
            hypos = hypoPath + "/" + file
            (semName, rest) = file.split(".", 1)
            # (segment,rest2) = rest.split(".", 1)
            # subdir = ""
            # subseg = ""
            # (site,rest) = semName.split("_", 1)
            # if site == "UKA":
            # subdir = "UKA"
            # subseg = "Segment" + segment
            # lowsite = string.lower(site)
            # labels=labelBase + "/" + subdir + "/" + semName + "/" + subseg
            # + "/" + lowsite + "_" + lab3dName
            # labels=labelBase + "/" + subdir + "/" + semName + "/" + subseg
            # + "/" + semName + "_" + lab3dName
            labels = labelBase + "/" + semName + "/" + semName + "_" + lab3dName

            print(hypos)
            print(labels)

            # run command

            print("*** SEGMENT #", file[:-3])
            if mode == "A":
                flag = "yes"
            else:
                flag = "no"
            cmd = "./MOTscore.pl %s %s %s" % (labels, hypos, flag)
            # print cmd
            out = getCommandOutput(cmd)
            print(out)

            # parse command output

            last = out[out.find("ABS TOTALS:"):]
            words = last.split()
            groundT += int(words[3])
            corr += int(words[5])
            fcorr += int(words[7])
            distance += int(words[9])
            miss += int(words[11])
            falseP += int(words[13])
            mismatch += int(words[15])

    ### calculate accumulated results

    print("\n")
    # print "SYSTEM: %s" % (hypoPath)
    # print "GROUND TRUTH %s" % (labelBase)

    if mode == "A":
        print(
            "********** TOTAL SUMMARY "
            "*********************************************")
        print("MOTP                    %.0fmm" % (distance / corr))
        print("MISS RATE               %.2f%%" % (100.0 * miss / groundT))
        print("FALSEPOS RATE           %.2f%%" % (100.0 * falseP / groundT))
        print("M/FP RATE (Loc Err > T) %.2f%%" % (100.0 * fcorr / groundT))
        print("A-MOTA                  %.2f%%" % (
                100.0 * (1.0 - (miss + falseP) / groundT)))
    else:
        print(
            "********** TOTAL SUMMARY "
            "*********************************************")
        print("MOTP         %.0fmm" % (distance / corr))
        print("MISSRATE     %.2f%%" % (100.0 * miss / groundT))
        print("FALSEPOSRATE %.2f%%" % (100.0 * falseP / groundT))
        print(
            "MISMATCHES   %d(%.2f%%)" % (mismatch, 100.0 * mismatch / groundT))
        print("MOTA         %.2f%%" % (
                100.0 * (1.0 - (miss + falseP + mismatch) / groundT)))


if __name__ == "__main__":
    main()
