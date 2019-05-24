#!/usr/bin/env python2

import argparse
import json
import logging
import sys

import numpy as np
from formatchecker import FormatChecker
from importers import MOT_groundtruth_import
from importers import MOT_hypo_import
from munkres import Munkres
from rect import Rect
from utilities import write_stderr_red

LOG = logging.getLogger(__name__)


class MOTEvaluation:

    def __init__(self, groundtruth, hypotheses, overlap_threshold):
        """Constructor """

        self.overlap_threshold_ = overlap_threshold
        """Bounding box overlap threshold"""

        self.munkres_inf_ = sys.maxsize
        """Not quite infinite number for Munkres algorithm"""

        self.sync_delta_ = 0.001
        """Maximum offset considered for a match of hypothesis and ground 
        truth"""

        self.groundtruth_ = groundtruth
        """Groundtruth. See groundtruth.json for a sample file"""

        if self.groundtruth_["class"] != "video":
            raise Exception("Ground truth is not of class \"video\"")

        self.hypotheses_ = hypotheses
        """Hypotheses. See hypotheses.json for a sample file"""

        if self.hypotheses_["class"] != "video":
            raise Exception("Hypotheses is not of class \"video\"")

        self.convertIDsToString()

        self.resetStatistics()

        # Set class and type for hypos and ground truths
        for f in self.hypotheses_["frames"]:
            for h in f["hypotheses"]:
                h["type"] = "hypothesis"
                h["class"] = "unevaluated"

        for f in self.groundtruth_["frames"]:
            for g in f["annotations"]:
                g["type"] = "groundtruth"
                g["class"] = "unevaluated"

        # List of dicts, containing ground truths and hypotheses for visual 
        # debugging
        self.visualDebugFrames_ = []

    def get_hypotheses_frame(self, timestamp):
        """Get list of hypotheses occuring chronologically close to ground 
        truth timestamp, but at most with time 
        difference self.sync_delta"""

        # Helper function for filter()
        def hypothesis_frame_chronologically_close(hypothesis):
            return abs(hypothesis["timestamp"] - timestamp) < self.sync_delta_

        # Hypotheses frames which are chronologically close to timestamp 
        # Use binary search, if this is to slow for you :)P
        hypotheses_frames = list(filter(hypothesis_frame_chronologically_close,
                                        self.hypotheses_["frames"]))

        # We expect at most one hypotheses timestamp.
        if len(hypotheses_frames) > 1:
            raise Exception(
                "> 1 hypotheses timestamps found for timestamp %f with sync "
                "delta %f" % (
                    timestamp, self.sync_delta_))

        if len(hypotheses_frames) == 0:
            #            write_stderr_red("Warning:", "No hypothesis 
            #            timestamp found for timestamp %f with sync delta
            #            %f" % (timestamp, self.sync_delta_))
            return {"hypotheses": []}  # empty list of hypos

        return hypotheses_frames[0]  # return first and only element of list

    def evaluate(self):
        """Compute MOTA metric from ground truth and hypotheses for all 
        frames."""

        frames = self.groundtruth_["frames"]
        for fr_idx, frame in enumerate(frames):
            self.evaluateFrame(frame)

    def evaluateFrame(self, frame):
        """Update statistics by evaluating a new frame."""

        timestamp = frame["timestamp"]
        groundtruths = frame["annotations"]
        hypotheses = self.get_hypotheses_frame(timestamp)["hypotheses"]

        visualDebugAnnotations = []

        # Save occuring ground truth ids
        for g in groundtruths:
            self.groundtruth_ids_.add(g["id"])

        # Save occuring hypothesis ids
        for h in hypotheses:
            self.hypothesis_ids_.add(h["id"])

        LOG.info("")
        LOG.info("Timestamp: %s" % timestamp)

        LOG.info("DIFF")
        LOG.info("DIFF Time %.2f" % timestamp)

        logstr = ["DIFF Mappings:"]
        for gt_id in sorted(self.mappings_.keys()):
            logstr.append("%s-%s" % (gt_id, self.mappings_[gt_id]))
        LOG.info(" ".join(logstr))

        # No need to evaluate this frame.
        if len(groundtruths) == 0 and len(hypotheses) == 0:
            LOG.info("No gt and hypos for this frame.")
            return

        LOG.info("GTs:")
        for groundtruth in groundtruths:
            LOG.info(Rect(groundtruth))

        LOG.info("Hypos:")
        for hypothesis in hypotheses:
            LOG.info(Rect(hypothesis))

        # PAPER STEP 1
        # Valid mappings skip Munkres algorithm, if both ground truth and 
        # hypo are found in this frame
        # We call these pairs correspondences and fill the list each frame.
        correspondences = {}  # truth id -> hypothesis id

        listofprints = []
        LOG.info("")
        LOG.info("STEP 1: KEEP CORRESPONDENCE")
        #            print "DIFF Keep correspondence"

        for gt_id in list(self.mappings_.keys()):
            groundtruth = [g for g in groundtruths if
                           g[
                               "id"] == gt_id]  # Get ground truths with 
            # given ground truth id in current frame
            if len(groundtruth) > 1:
                LOG.warning("found %d > 1 ground truth tracks for id %s",
                            len(groundtruth), gt_id)
            elif len(groundtruth) < 1:
                continue

            hypothesis = [h for h in hypotheses if
                          h["id"] == self.mappings_[
                              gt_id]]  # Get hypothesis with hypothesis id 
            # according to mapping
            assert len(hypothesis) <= 1
            if len(hypothesis) != 1:
                continue

            # Hypothesis found for known mapping
            # Check hypothesis for overlap
            overlap = Rect(groundtruth[0]).overlap(Rect(hypothesis[0]))
            if overlap >= self.overlap_threshold_:
                LOG.info("Keeping correspondence between %s and %s" % (
                    groundtruth[0]["id"], hypothesis[0]["id"]))
                #                    print "DIFF Keep corr %s %s %.2f" % (
                #                    groundtruth[0]["id"], hypothesis[0]["id"], 
                #                    Rect(groundtruth[0]).overlap(Rect(
                #                    hypothesis[0])))
                listofprints.append("DIFF Keep corr %s %s %.2f" % (
                    groundtruth[0]["id"], hypothesis[0]["id"],
                    Rect(groundtruth[0]).overlap(Rect(hypothesis[0]))))
                correspondences[gt_id] = hypothesis[0]["id"]
                self.total_overlap_ += overlap

        for p in sorted(listofprints):
            LOG.info(p)

        # PAPER STEP 2
        LOG.info("")
        LOG.info("STEP 2: FIND CORRESPONDENCE")

        # Fill hungarian matrix with +inf
        munkres_matrix = [[self.munkres_inf_ for i in range(len(hypotheses))]
                          for j in
                          range(len(groundtruths))]  # TODO make square matrix

        # Find correspondences
        for i in range(len(groundtruths)):
            groundtruth = groundtruths[i]

            # Skip groundtruth with correspondence from mapping
            if groundtruth["id"] in list(correspondences.keys()):
                LOG.info(
                    "Groundtruth %s already in correspondence" % groundtruth[
                        "id"])
                continue

            # Fill hungarian matrix with distance between gts and hypos
            for j in range(len(hypotheses)):
                hypothesis = hypotheses[j]

                # Skip hypotheses with correspondence from mapping
                if hypothesis["id"] in list(correspondences.values()):
                    LOG.info(
                        "Hypothesis %s already in correspondence" % hypothesis[
                            "id"])
                    continue

                rect_groundtruth = Rect(groundtruth)
                rect_hypothesis = Rect(hypothesis)
                overlap = rect_groundtruth.overlap(rect_hypothesis)

                if overlap >= self.overlap_threshold_:
                    #                        print "Fill Hungarian", 
                    #                        rect_groundtruth, 
                    #                        rect_hypothesis, overlap
                    munkres_matrix[i][j] = 1 / overlap
                    LOG.info("DIFF candidate %s %s %.2f" % (
                        groundtruth["id"], hypothesis["id"], overlap))

        # Do the Munkres
        LOG.debug(munkres_matrix)

        # Only run munkres on non-empty matrix
        if len(munkres_matrix) > 0:
            m = Munkres()
            indices = m.compute(munkres_matrix)
        else:
            LOG.info(
                "No need to run Hungarian with %d ground truths and %d "
                "hypothesis." % (
                    len(groundtruths), len(hypotheses)))
            indices = []
        LOG.info(indices)

        correspondencelist = []
        mismatcheslist = []

        for gt_index, hypo_index in indices:

            # Skip invalid self.mappings_
            # Check for max float distance matches (since Hungarian returns 
            # complete mapping)
            if (munkres_matrix[gt_index][
                hypo_index] == self.munkres_inf_):  # NO correspondence <=> 
                # overlap >= thresh
                continue

            gt_id = groundtruths[gt_index]["id"]
            hypo_id = hypotheses[hypo_index]["id"]

            # Assert no known mappings have been added to hungarian, 
            # since keep correspondence should have considered
            # this case.
            if gt_id in self.mappings_:
                assert self.mappings_[gt_id] != hypo_id

                # Add to correspondences
            LOG.info("Correspondence found: %s and %s (overlap: %f)" % (
                gt_id, hypo_id, 1.0 / munkres_matrix[gt_index][hypo_index]))
            #                correspondencelist.append("DIFF correspondence 
            #                %s %s %.2f" % (gt_id, hypo_id, 
            #                1.0 / munkres_matrix[gt_index][hypo_index]))
            correspondencelist.append(
                "DIFF correspondence %s %s" % (gt_id, hypo_id))
            correspondences[gt_id] = hypo_id
            self.total_overlap_ += 1.0 / munkres_matrix[gt_index][hypo_index]

            # Count "recoverable" and "non-recoverable" mismatches
            # "recoverable" mismatches
            if gt_id in self.gt_map_ and self.gt_map_[gt_id] != hypo_id and \
                    not \
                            groundtruths[gt_index].get("dco",
                                                       False):
                LOG.info(
                    "Look ma! We got a recoverable mismatch over here! ("
                    "%s-%s) -> (%s-%s)" % (
                        gt_id, self.gt_map_[gt_id], gt_id, hypo_id))
                self.recoverable_mismatches_ += 1

            # "non-recoverable" mismatches
            if hypo_id in self.hypo_map_ and self.hypo_map_[hypo_id] != gt_id:
                # Do not count non-recoverable mismatch, if both old ground 
                # truth and current ground truth are DCO.
                old_gt_id = self.hypo_map_[hypo_id]
                old_gt_dco = [g for g in groundtruths if
                              g["id"] == old_gt_id and g.get("dco", False)]

                assert len(old_gt_dco) <= 1;
                if not (groundtruths[gt_index].get("dco", False) and len(
                        old_gt_dco) == 1):
                    LOG.info(
                        "Look ma! We got a non-recoverable mismatch over "
                        "here! (%s-%s) -> (%s-%s)" % (
                            self.hypo_map_[hypo_id], hypo_id, gt_id, hypo_id))
                    self.non_recoverable_mismatches_ += 1

            # Update yin-yang maps                    
            self.gt_map_[gt_id] = hypo_id
            self.hypo_map_[hypo_id] = gt_id

            # Correspondence contradicts previous mapping. Mark and count as 
            # mismatch, if ground truth is not a DCO
            # Iterate over all gt-hypo pairs of mapping, since we have to 
            # perform a two way check:
            # Correspondence: A-1
            # Mapping: A-2, B-1
            # We have to detect both forms of conflicts
            for mapping_gt_id, mapping_hypo_id in list(self.mappings_.items()):

                # CAVE: Other than in perl script:
                # Do not consider for mismatch, if both old gt and new gt are
                # DCO
                gt_with_mapping_gt_id_dco = [g for g in groundtruths if
                                             g["id"] == mapping_gt_id and g.get(
                                                 "dco", False)]
                if len(gt_with_mapping_gt_id_dco) == 1 and groundtruths[
                    gt_index].get("dco", False):
                    LOG.info(
                        "Ground truths %s and %s are DCO. Not considering for "
                        "mismatch." % (
                            mapping_gt_id, gt_id))
                #                    print "DIFF DCO %s" % (gt_id), 
                #                    groundtruths[gt_index]

                else:
                    # Look ma, we got a conflict over here!
                    # New hypothesis for mapped ground truth found
                    if (mapping_gt_id == gt_id and mapping_hypo_id != hypo_id) \
                            or (
                            mapping_gt_id != gt_id and mapping_hypo_id ==
                            hypo_id):
                        LOG.info(
                            "Correspondence %s-%s contradicts mapping %s-%s. "
                            "Counting as mismatch and updating "
                            "mapping." % (
                                gt_id, hypo_id, mapping_gt_id, mapping_hypo_id))
                        mismatcheslist.append(
                            "DIFF Mismatch %s-%s -> %s-%s" % (
                                mapping_gt_id, mapping_hypo_id, gt_id, hypo_id))
                        self.mismatches_ = self.mismatches_ + 1

                        # find groundtruth and hypothesis with given ids
                        g = [g for g in groundtruths if g["id"] == gt_id]
                        h = [h for h in hypotheses if h["id"] == hypo_id]

                        # assert(len(g) == 1)
                        if len(g) != 1:
                            LOG.warning('more than one gt: %s', str(g))
                        assert (len(h) == 1), \
                            "There're {} hypos with same ID {} in the {}".format(
                                len(h), hypo_id, frame['im_path'])

                        g = g[0]
                        h = h[0]

                        g["class"] = "mismatch"
                        h["class"] = "mismatch"

                        visualDebugAnnotations.append(g)
                        visualDebugAnnotations.append(h)

                        # mapping will be updated after loop
                        del self.mappings_[mapping_gt_id]

            #                print "YIN: %d %d" % (
            #                self.recoverable_mismatches_, 
            #                self.non_recoverable_mismatches_)
            #                assert(self.recoverable_mismatches_ + 
            #                self.non_recoverable_mismatches_ == 
            #                self.mismatches_)
            if (
                    self.recoverable_mismatches_ +
                    self.non_recoverable_mismatches_ != self.mismatches_):
                LOG.info("Look, mismatches differ: g %d b %d  other %d" % (
                    self.recoverable_mismatches_,
                    self.non_recoverable_mismatches_, self.mismatches_))
                LOG.info(self.gt_map_)
                LOG.info(self.hypo_map_)

            # Save (overwrite) mapping even if ground truth is dco
            self.mappings_[gt_id] = hypo_id  # Update mapping

        # Sorted DIFF output
        for c in sorted(correspondencelist):
            LOG.info(c)

        for m in sorted(mismatcheslist):
            LOG.info(m)

        # Visual debug
        for g in groundtruths:
            if g["class"] != "mismatch" and g["id"] in list(
                    correspondences.keys()):
                g["class"] = "correspondence"
                visualDebugAnnotations.append(g)

        for h in hypotheses:
            if h["class"] != "mismatch" and h["id"] in list(
                    correspondences.values()):
                h["class"] = "correspondence"
                visualDebugAnnotations.append(h)

        # TODO get overlap ratio
        # Print out correspondences
        #            for gt_id, hypo_id in correspondences.items():
        #                print "Correspondence: %s-%s" % (gt_id, hypo_id)

        # PAPER STEP 4
        # Count miss, when groundtruth has no correspondence and is not dco
        for groundtruth in groundtruths:
            LOG.info("DCO:", groundtruth)
            if groundtruth["id"] not in list(
                    correspondences.keys()) and groundtruth.get("dco",
                                                                False) != True:
                LOG.info("Miss: %s" % groundtruth["id"])
                LOG.info("DEBUGMISS: %.2f" % timestamp)
                LOG.info("DIFF Miss %s" % groundtruth["id"])
                groundtruth["class"] = "miss"
                visualDebugAnnotations.append(groundtruth)
                self.misses_ += 1

        # Count false positives
        for hypothesis in hypotheses:
            if hypothesis["id"] not in list(correspondences.values()):
                LOG.info("False positive: %s" % hypothesis["id"])
                LOG.info("DIFF False positive %s" % hypothesis["id"])
                self.false_positives_ += 1
                visualDebugAnnotations.append(hypothesis)
                hypothesis["class"] = "false positive"


        self.total_correspondences_ += len(correspondences)

        self.total_groundtruths_ += len(
            groundtruths)  # Number of objects (ground truths) in current frame

        visualDebugFrame = {
            "timestamp": timestamp,
            "class": frame["class"],
            "annotations": visualDebugAnnotations
        }
        if "num" in frame:
            visualDebugFrame["num"] = frame["num"]

        self.visualDebugFrames_.append(visualDebugFrame)

    @staticmethod
    def calcMOTA(abs_stats):
        num_gt = abs_stats['ground truths']
        if num_gt == 0:
            write_stderr_red("Warning",
                             "No ground truth. MOTA calculation not possible")
            return 0.0
        return 1.0 - float(
            abs_stats['misses'] + abs_stats['false positives'] + abs_stats[
                'mismatches']) / num_gt

    @staticmethod
    def calcMOTP(abs_stats):
        num_corr = abs_stats['correspondences']
        if num_corr == 0:
            write_stderr_red("Warning",
                             "No ground truth. MOTA calculation not possible")
            return 0.0
        return float(abs_stats['total overlap']) / num_corr

    @staticmethod
    def calcRelativeStatistics(abs_stats):
        gt = abs_stats['ground truths']
        num_gt_tracks = abs_stats['lonely ground truth tracks'] + abs_stats[
            'covered ground truth tracks']

        return {
            "MOTA": MOTEvaluation.calcMOTA(abs_stats),
            "MOTP": MOTEvaluation.calcMOTP(abs_stats),
            "miss rate": float(abs_stats['misses']) / gt,
            "false positive rate": float(abs_stats['false positives']) / gt,
            "mismatch rate": float(abs_stats['mismatches']) / gt,
            "recoverable mismatch rate": float(
                abs_stats['recoverable mismatches']) / gt,
            "non-recoverable mismatch rate": float(
                abs_stats['non-recoverable mismatches']) / gt,
            "track precision": float(abs_stats['covering hypothesis tracks']) /
                               abs_stats['hypothesis tracks'] if
            abs_stats['hypothesis tracks'] != 0 else 0.0,
            "track recall": float(
                abs_stats[
                    'covered ground truth tracks']) / num_gt_tracks if
            num_gt_tracks != 0 else 0.0,
        }

    def getMOTA(self):
        mota = 0.0
        if self.total_groundtruths_ == 0:
            write_stderr_red("Warning",
                             "No ground truth. MOTA calculation not possible")
        #            raise("No ground truth. MOTA calculation not possible")
        else:
            mota = 1.0 - float(
                self.misses_ + self.false_positives_ + self.mismatches_) / \
                   float(
                       self.total_groundtruths_)

        return mota

    def getMOTP(self):
        motp = 0.0
        if self.total_correspondences_ == 0:
            write_stderr_red("Warning",
                             "No correspondences found. MOTP calculation not "
                             "possible")
        #            raise("No correspondence found. MOTP calculation not 
        #            possible")
        else:
            motp = self.total_overlap_ / self.total_correspondences_
        return motp

    def getAbsoluteStatistics(self):
        lonely_ground_truths = self.groundtruth_ids_ - set(self.gt_map_.keys())
        covered_ground_truths = self.groundtruth_ids_ & set(self.gt_map_.keys())
        lonely_hypotheses = self.hypothesis_ids_ - set(self.hypo_map_.keys())

        return {
            "ground truths": self.total_groundtruths_,
            "false positives": self.false_positives_,
            "misses": self.misses_,
            "mismatches": self.mismatches_,
            "recoverable mismatches": self.recoverable_mismatches_,
            "non-recoverable mismatches": self.non_recoverable_mismatches_,
            "correspondences": self.total_correspondences_,
            "total overlap": self.total_overlap_,
            "lonely ground truth tracks": len(lonely_ground_truths),
            "covered ground truth tracks": len(covered_ground_truths),
            "lonely hypothesis tracks": len(lonely_hypotheses),
            "ground truth tracks": len(self.groundtruth_ids_),
            "hypothesis tracks": len(self.hypothesis_ids_),
            "covering hypothesis tracks": len(list(self.hypo_map_.keys()))
        }

    def getRelativeStatistics(self):
        gt = self.total_groundtruths_
        covered_ground_truths = self.groundtruth_ids_ & set(self.gt_map_.keys())
        lonely_hypotheses = self.hypothesis_ids_ - set(self.hypo_map_.keys())

        return {
            "MOTA": self.getMOTA(),
            "MOTP": self.getMOTP(),
            "miss rate": float(self.misses_) / gt,
            "false positive rate": float(self.false_positives_) / gt,
            "mismatch rate": float(self.mismatches_) / gt,
            "recoverable mismatch rate": float(
                self.recoverable_mismatches_) / gt,
            "non-recoverable mismatch rate": float(
                self.non_recoverable_mismatches_) / gt,
            "track precision": float(len(list(self.hypo_map_.keys()))) / len(
                self.hypothesis_ids_) if len(
                self.hypothesis_ids_) != 0 else 0.0,
            "track recall": float(len(list(self.gt_map_.keys()))) / len(
                self.groundtruth_ids_) if len(
                self.groundtruth_ids_) != 0 else 0.0,
        }

    def printTrackStatistics(self):
        # Lonely ground truths (no single correspondence)
        lonely_ground_truths = self.groundtruth_ids_ - set(self.gt_map_.keys())
        print("Lonely ground truth tracks %d" % len(lonely_ground_truths))
        print("Total ground truth tracks  %d" % len(self.groundtruth_ids_))
        #        print "    ", lonely_ground_truths

        # Dirty false positive tracks (no single correspondence)
        lonely_hypotheses = self.hypothesis_ids_ - set(self.hypo_map_.keys())
        print("Lonely hypothesis tracks %d" % len(lonely_hypotheses))
        print("Total hypothesis tracks  %d" % len(self.hypothesis_ids_))

    #        print "    ", lonely_hypotheses

    def printResults(self):
        """Print out results"""
        # Additional statistics
        print("Ground truths               %d" % self.total_groundtruths_)
        print("False positives             %d" % self.false_positives_)
        print("Misses                      %d" % self.misses_)
        print("Mismatches                  %d" % self.mismatches_)
        print("Recoverable mismatches      %d" % self.recoverable_mismatches_)
        print(
            "Non recoverable mismatches  %d" % self.non_recoverable_mismatches_)
        print("Correspondences             %d" % self.total_correspondences_)
        print("")
        print("MOTP", self.getMOTP())
        print("MOTA", self.getMOTA())

    def printLegacyFormat(self):
        """Print out as expected by score_all tool."""
        # TODO implement correspondences, faildCorrespondences and 
        #  overlap_ratio 
        print(
            "DIFF TOTALS: groundT %.0f\t miss %.0f\t falseP %.0f\t mismatch "
            "%0.f" % (
                self.total_groundtruths_, self.misses_, self.false_positives_,
                self.mismatches_))
        print(
            "ABS TOTALS: groundT %.0f\t corr %.0f\t failedCorr %.0f\t "
            "overlap_ratio %.2f\t miss %.0f\t falseP %.0f\t "
            "mismatch %0.f" % (
                self.total_groundtruths_, 0, 0, 0, self.misses_,
                self.false_positives_, self.mismatches_))

    def convertIDsToString(self):
        for f in self.groundtruth_["frames"]:
            for g in f["annotations"]:
                g["id"] = str(g.get("id", '__missing_id__'))

        for f in self.hypotheses_["frames"]:
            for h in f["hypotheses"]:
                h["id"] = str(h["id"])

    def getVisualDebug(self):
        fileitem = {
            'filename': self.groundtruth_["filename"],
            'class': self.groundtruth_["class"],
            'frames': self.visualDebugFrames_
        }
        return [fileitem]

    def resetMapping(self):
        """Reset mapping. Useful for loading new ground truth and hypo and 
        not counting shot-boundary caused 
        mismatches."""
        self.mappings_ = {}  # Mappings from ground truth id to hypothesis 
        # id, as described in paper: M_t (initial 
        # M_0 empty)

        # Helper dicts for "recoverable" and "non-recoverable" mismatch 
        # detection aka Yin Yang
        self.gt_map_ = {}  # save most recent hypothesis id for each 
        # groundtruth id. Only updates, no deletions of keys
        self.hypo_map_ = {}  # save move recent groundtruth id for each 
        # hypothesis id. Only updates, no deletions of 
        # keys.

    def resetStatistics(self):
        """Reset counters and mapping."""
        self.resetMapping()

        # yin-yang
        self.recoverable_mismatches_ = 0
        self.non_recoverable_mismatches_ = 0

        # MOTA related
        self.mismatches_ = 0
        self.misses_ = 0
        self.false_positives_ = 0
        self.total_groundtruths_ = 0
        self.total_overlap_ = 0.0
        self.total_correspondences_ = 0

        self.groundtruth_ids_ = set()
        self.hypothesis_ids_ = set()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                        description='3D Tracking BDD Format Evaluation',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--groundtruth', required=True)
    parser.add_argument('-b', '--hypothesis', required=True)
    parser.add_argument('-c', '--check_format', action="store_true",
                        default=True)
    parser.add_argument('-v', '--visual_debug_file')
    parser.add_argument('-i', '--iou', default=0.2, type=float,
                        help='iou threshold')
    args = parser.parse_args()

    # Load ground truth according to format
    # Assume MOT format, if non-json
    gt = open(args.groundtruth)  # gt file
    if args.groundtruth.endswith(".json"):
        groundtruth = json.load(gt)[0]
    else:
        groundtruth = MOT_groundtruth_import(gt.readlines())
    gt.close()

    # Load MOT format files
    hypo = open(args.hypothesis)  # hypo file
    if args.hypothesis.endswith(".json"):
        hypotheses = json.load(hypo)[0]
    else:
        hypotheses = MOT_hypo_import(hypo.readlines())
    hypo.close()

    evaluator = MOTEvaluation(groundtruth, hypotheses, args.iou)

    if (args.check_format):
        formatChecker = FormatChecker(groundtruth, hypotheses)
        success = formatChecker.checkForExistingIDs()
        success |= formatChecker.checkForAmbiguousIDs()
        success |= formatChecker.checkForCompleteness()

        if not success:
            write_stderr_red("Error:",
                             "Stopping. Fix ids first. Evaluating with broken "
                             "data does not make sense!\n    File: "
                             "%s" % args.groundtruth)
            sys.exit()

    evaluator.evaluate()
    print("Track statistics")
    evaluator.printTrackStatistics()
    print()
    print("Results")
    evaluator.printResults()
    #    evaluator.printLegacyFormat()

    #    print json.dumps(evaluator.getAbsoluteStatistics(), indent=4,
    #    sort_keys=True)
    #    print json.dumps(evaluator.getRelativeStatistics(), indent=4,
    #    sort_keys=True)

    if (args.visual_debug_file):
        with open(args.visual_debug_file, 'w') as fp:
            json.dump(evaluator.getVisualDebug(), fp, indent=4)
