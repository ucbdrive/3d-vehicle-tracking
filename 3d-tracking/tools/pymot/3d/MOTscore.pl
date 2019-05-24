#!/usr/bin/perl -w

require "Munkre.pl";

use strict;

# Constants

# Parameters
my $truthfile = shift || die "First Parameter: 3D-Ground Truth file"; 
my $hypofile = shift || die "Second Parameter: Hypothesis file";
my $A_MOTA = shift || die "Third Parameter: Count mismatches? <yes/no>";
my $THRESHOLD = 500.0;
my $HIGHEST_FLOAT = 5.0*(10**17); # This is used when determining optimal mapping.
my $SYNC_DELTA = 0.5; # This is the maximum offest to match hypo and ground truth frames

my @truthBuffer;
my @hypoBuffer;
my @hypotimes;

my $truthstamp = 0.0;
my @truthid; my @truthx; my @truthy;
my @hypoid; my @hypox; my @hypoy;


my %correspondence;
my %mapping; # to detect mismatches after misses
my %groundtruths; #keeps a list of all encountered ground truths

my %avgdist;   # Average distance
my %corr;      # Number of correspondences
my %misses;       # Number of misses 
my %mismatches; # Number of ID mismatches
my $falseP = 0;  # Number of false positives
my $fCorr = 0;  # Number of failed correspondences

&init();

&loadTruth();
&loadHypo();
foreach my $truthLine (@truthBuffer){
    &findTruth($truthLine);
    if ( &findHypo() == 0 ) {
	print "Hypo has no more data. Tracker interrupted?\n";
	last; 
    }
    
#  # Output all known data
#  print "$truthstamp - ";
#  for( my $i=0; $i<=$#truthid; $i++ ) {
#    print " ($truthid[$i]: $truthx[$i] $truthy[$i]) ";
#  }
#  print " --- ";
#  for( my $i=0; $i<=$#hypoid; $i++ ) {
#    print " ($hypoid[$i]: $hypox[$i] $hypoy[$i]) ";
#  }
#  print "\n";
    
    &keepCorrespondence();
    &findCorrespondence();
    &errorcomputation();
}

&summary();
&finish();


###
# Init
sub init {
  %correspondence = ();
  %mapping = ();
  %avgdist = ();
  %corr = ();
  %misses = ();
  %mismatches = ();
}

###
# und tschuess
sub finish {
}

###
# Load in the ground truths
sub loadTruth {
    open(TRUTH, "<$truthfile") || die "No ground truth: $!";
    while(<TRUTH>) {
	chomp;
	push @truthBuffer, $_;
    }
    close(TRUTH);
}

###
# Load in the ground hypos
sub loadHypo {
    open(HYPO, "<$hypofile") || die "No hypotheses: $!";
    while(<HYPO>) {
	chomp;
	push @hypoBuffer, $_;
	s/^(\d+(\.\d+)?)\s*//;
	push @hypotimes, $1;
    }
    close(HYPO);
    #for( my $i=0; $i<=$#hypotimes; $i++ ) {
	#print "$i: ";
	#print "$hypotimes[$i]\n";
    #}
}

###
# Parse a line of the ground truth file and store in appropriate global arrays
sub findTruth {
    my ($curtruth) = @_;
    $_ = $curtruth; 
    s/^(\d+(\.\d+)?)\s*//;
    $truthstamp = $1;
    
    @truthid = (); @truthx = (); @truthy = (); $#truthid = -1;
    while( s/^(\w+_?\w*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*// ) {
	push @truthid, $1;
	push @truthx, $2;
	push @truthy, $3;
	# z is ignored
	
	if ( !exists $groundtruths{$1}) {$groundtruths{$1} = 1;}
    }
}

sub findBestHypoLine{
    my ($stamp) = @_;
    my $res=0;
    if($#hypotimes<0){
	return "0\n";
    }
    if($stamp<=$hypotimes[0]){
	$res=0;
    }elsif($stamp>=$hypotimes[$#hypotimes-1]){
	$res=$#hypotimes-1;
    }else{
	$res=binarySearch($stamp,0,$#hypotimes-1);
  }
  return $hypoBuffer[$res];
}

sub binarySearch{
    my ($t,$i1,$i2) = @_;
    if($i1==$i2){
	return $i1;
    }elsif ($i1==$i2-1){
	if (($t-$hypotimes[$i1]) <= ($hypotimes[$i2]-$t)){
	    return $i1;
	}
	else{
	    return $i2;
	}
    }else{
	my $mid=int(($i2+$i1)/2);
	if($t>$hypotimes[$mid]){
	    return binarySearch($t,$mid,$i2);
	}else{
	    return binarySearch($t,$i1,$mid);
	}
    }
}

###
# Find Hypo data time synchronous to ground truth and store globally
sub findHypo {
    my $besthypo;
    my $beststamp;
    my $rest;
    
    # Clear these lists here! because we might break off later without filling them...
    @hypoid = (); @hypox = (); @hypoy = (); $#hypoid = -1;
    
    # Search for a matching entry here. Look for the closest hypo within +-0.5secs
    $besthypo = findBestHypoLine($truthstamp);
    $_ = $besthypo;
    s/^(\d+(\.\d+)?)\s*//;
    $beststamp = $1;
    $rest = $_."\n";
    if ( ($beststamp < $truthstamp - $SYNC_DELTA) || ($beststamp >= $truthstamp + $SYNC_DELTA) ) {
	$beststamp = 0.0;
    }
    
    if ( $beststamp == 0.0 ) { 
	# if no match could be found, this timeframe is counted as a complete miss:
	# nothing is pushed into the hypoid list and we return success
	print "No match: $truthstamp\n";
	close(HYPO);
	return 1;
    }
    #print "Match: $truthstamp $beststamp $rest\n";
    $_ = $rest;
    while( s/^(\S+)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*// ) {
	push @hypoid, $1;
	push @hypox, $2;
	push @hypoy, $3;
	# z is ignored
    }
    return 1;
}

###
# For free ground truth persons, look for corresponding Hypotheses.
# Remember these Correspondences!
sub findCorrespondence {
  my @matrixdis = ();
  my %listtruths;
  my %listhypos;
  my %Rlisttruths;
  my %Rlisthypos;
  my $n=0;
  my $m=0;
  
  for( my $i=0; $i<=$#truthid; $i++ ) {
      my $TID = $truthid[$i];
      if ( !exists $correspondence{$TID} ) {
	  # ground truth without correspondence
	  #print "Searching correspondence for Truth $TID\n";
	  # search through all Hypos
	HYPO: for( my $hypo=0; $hypo<=$#hypoid; $hypo++ ) {
	    # is hypo still available?
	    foreach my $co (values (%correspondence)) {
		if ( $co eq $hypoid[$hypo] ) {
		    #print "Hypo $hypoid[$hypo] already matched\n";
		    next HYPO;
		}
	    }
	    # Distance small enough 
	    my $currdistance = distance($i, $hypo);
	    if ($currdistance <= $THRESHOLD ) {
		my $Tind;
		if (exists $listtruths{$TID}) {
		    $Tind = $listtruths{$TID};
		} else {
		    push @matrixdis, [];
		    for (my $col=0; $col<$m; $col++) {
			push @{$matrixdis[$n]}, $HIGHEST_FLOAT;
		    }
		    $Tind = $n;
		    $listtruths{$TID} = $Tind;
		    $Rlisttruths{$Tind} = $TID;
		    $n++;
		}
		my $Hind;
		if (exists $listhypos{$hypoid[$hypo]}) {
		    $Hind = $listhypos{$hypoid[$hypo]};
		} else {
		    for (my $row=0; $row<$n; $row++) {
			push @{$matrixdis[$row]}, $HIGHEST_FLOAT;
		    }
		    $Hind = $m;
		    $listhypos{$hypoid[$hypo]} = $Hind;
		    $Rlisthypos{$Hind} = $hypoid[$hypo];
		    $m++;
		}
		$matrixdis[$Tind][$Hind] = $currdistance;
		#print "for truth $TID, new hypo-candidate $hypoid[$hypo], ".
		    #"distance=$currdistance\n";
	    } else {
		#print "for truth $TID, the hypo $hypo doesn't fit, distance=".
		    #"$currdistance\n";
	    }
	}
      }
  }
  
  ## Make the optimal mapping based on Munkre's algorithm
  my @Map;
  if ($n>0 && $m>0) {
      @Map = munkre(@matrixdis);
  }
  for (my $i=0; $i<$n; $i++) {
      for (my $j=0; $j<$m; $j++) {
	  if ($Map[$i][$j] == 1) {
	      if ($matrixdis[$i][$j] <= $THRESHOLD) {# Do not map if dist > T
		  # make the correspondence
		  my $maptruth = $Rlisttruths{$i};
		  my $maphypo = $Rlisthypos{$j};
		  
		  #print "New Correspondence: truth $maptruth = hypo $maphypo, ".
		  #"distance=$matrixdis[$i][$j]\n";
		  $correspondence{$maptruth} = $maphypo;
		  
		  # check for mismatches and update the mapping
		  my @conflictList = ();
		  foreach my $XTID (keys (%mapping)) {
		      if ( defined $mapping{$XTID} ) {
			  if ( (($XTID eq $maptruth) && ($mapping{$XTID} ne $maphypo))
			       || (($XTID ne $maptruth) && ($mapping{$XTID} eq $maphypo))) {
			      # get rid of the old mapping
			      push @conflictList, $XTID;
			      # and count mismatches
			      if ( exists $mismatches{$maptruth} ) {
				  $mismatches{$maptruth}++;
			      } else {
				  $mismatches{$maptruth} = 1;
			      }
			      ####print "$truthstamp: Mismatch ($XTID <-> $mapping{$XTID}) -> ($maptruth <-> $maphypo), ".
				  ####"mismatches{$maptruth} = $mismatches{$maptruth}\n";
			  }
		      }
		  }
		  if (!defined $mapping{$maptruth}) {
		      ####print "$truthstamp: mapping{$maptruth} = $maphypo\n";
		  }
		  foreach my $CTID (@conflictList) {
		      undef $mapping{$CTID};
		  }
		  $mapping{$maptruth} = $maphypo;
	      }
	  }
      }
  }
}

###
# If a correspondence is still valid, keep it
sub keepCorrespondence {
    %correspondence = ();
    foreach my $TID (keys (%mapping)) {
	if ( defined $mapping{$TID} ) {
	    my $truth = truthLookup($TID);
	    my $hypo = hypoLookup($mapping{$TID});
	    if ( ($truth != -1) && ($hypo != -1)) { 
		if (distance( $truth, $hypo ) <= $THRESHOLD ) {
		    #print "Keep Correspondence Ground Truth $TID with Hypothesis $mapping{$TID}\n";
		    $correspondence{$TID} = $mapping{$TID};
		} else {
		    #print "Correspondence $TID <-> $mapping{$TID} no longer valid\n";
		}
	    }
	}
    }
}

###
# Look for index of truthid in array
sub truthLookup {
  my ($searchtruth) = @_;
  for( my $i=0; $i<=$#truthid; $i++ ) {
    if ( $truthid[$i] eq $searchtruth ) {
      return $i;
    }
  }
  return -1;
}

###
# Look for index of hypoid in array
sub hypoLookup {
  my ($searchhypo) = @_;
  for( my $i=0; $i<=$#hypoid; $i++ ) {
    if ( $hypoid[$i] eq $searchhypo ) {
      return $i;
    }
  }
  return -1;
}

###
# Calculate Euclidian Distance between #truth and Hypo
sub distance {
  my ($truth, $hypo) = @_;
  return sqrt ( (($truthx[$truth] - $hypox[$hypo])*
                 ($truthx[$truth] - $hypox[$hypo])) +
                (($truthy[$truth] - $hypoy[$hypo])*
                 ($truthy[$truth] - $hypoy[$hypo])) );
}

###
# Compute: distance, misses, falseP
sub errorcomputation {
  #my $oldmisses = eval join("+", @misses);
  my $actCorrs = 0;
  for(my $i=0; $i<=$#truthid; $i++ ) {
    my $TID = $truthid[$i];
    if ( exists $correspondence{$TID} ) {
      # Distance measurable
      my $dis = distance( $i, hypoLookup($correspondence{$TID}) );
      #print "error computation: Distance Truth $TID and hypo $correspondence{$TID} = $dis\n";
      if ( !exists $avgdist{$TID} ) {
        $avgdist{$TID} = $dis;
        $corr{$TID} = 1;
      } else {
        $avgdist{$TID} = ( $avgdist{$TID} * $corr{$TID} + $dis ) /
            ++$corr{$TID};
      }
      #print "avg = $avgdist{$TID}, corr{$TID} = $corr{$TID}\n";
      $actCorrs++;
    } else {
      # miss
      if ( !exists $misses{$TID} ) {
	$misses{$TID} = 1;
      } else {
	$misses{$TID}++;
      }
      #print "misses{$TID} = $misses{$TID}\n";
    }
  }
  #print MISSES "$frame " . ((eval join("+",@misses)) - $oldmisses) ."\n";

  #my $oldfalseP = $falseP;
  # uebrige hypo suchen
  for( my $hypo=0; $hypo<=$#hypoid; $hypo++ ) {  
    my $tmpfp = 1;
    foreach my $co (values (%correspondence)) {
      if ( $co eq $hypoid[$hypo] ) {
        $tmpfp = 0;
      }
    }
    $falseP += $tmpfp;
  }
  #print "falseP=$falseP\n";
  #print FALSEP "$frame " . ($falseP-$oldfalseP) ."\n";
  
  my $minGTH = $#truthid+1;
  if ($#hypoid+1 < $minGTH) {
      $minGTH = $#hypoid+1;
  }
  
  $fCorr += ($minGTH - $actCorrs);
  #print "fCorr = $minGTH - $actCorrs = $fCorr\n";
}

###
# die wichtigen Werte loggen
sub summary {
  print "****************SUMMARY*********************\n";

  # complete missing hash entries
  print "Groundtruth:";
  foreach my $TID (keys (%groundtruths)) {
    if ( !exists $avgdist{$TID} ) {$avgdist{$TID} = 0;}
    if ( !exists $corr{$TID} ) {$corr{$TID} = 0;}
    if ( !exists $misses{$TID} ) {$misses{$TID} = 0;}
    if ( !exists $mismatches{$TID} ) {$mismatches{$TID} = 0;}
    print " $TID";
  }
  print "\n";

  my $alldist = 0;
  my $allavgdist = 0;
  my $allground = 0;
  my $allcorr = 0;
  print "avgdist:";
  foreach my $TID (keys (%groundtruths)) {
    print " $avgdist{$TID}";
    $alldist+=$avgdist{$TID}*($corr{$TID});
    $allcorr+=$corr{$TID};
    $allground+=$corr{$TID}+$misses{$TID};
  }
  print "\n";
  if ( $allcorr != 0 ) {
    $allavgdist = $alldist/$allcorr;
  } else {
    $allavgdist = -1;
  }

  print "corr:";
  foreach my $TID (keys (%groundtruths)) {
    print " $corr{$TID}";
  }
  print "\n";

  my $allmisses = 0;
  my $allavgmisses = 0;
  print "misses:";
  foreach my $TID (keys (%groundtruths)) {
    print " $misses{$TID} (" . ($misses{$TID}/($corr{$TID}+$misses{$TID})*100) .
        "%)";
    $allmisses += $misses{$TID};
  }
  print "\n";
  $allavgmisses = $allmisses/$allground;

  my $allmismatch = 0;
  my $allavgmismatch = 0;
  print "mismatches:";
  foreach my $TID (keys (%groundtruths)) {
    print " $mismatches{$TID}";
    $allmismatch += $mismatches{$TID};
  }
  print "\n";  
  $allavgmismatch = $allmismatch / $allground;

  print "falseP: $falseP\n";
  my $allfp;
  my $allavgfp;
  $allfp = $falseP;
  $allavgfp = $allfp / $allground;

  print "failed Corr: $fCorr\n";
  my $allfCorr;
  my $allavgfCorr;
  $allfCorr = $fCorr;
  $allavgfCorr = $allfCorr / $allground;

  ## Modified to ignore mismatches on demand
  my $accuracy;
  my $a_accuracy;
  
  $accuracy = 1 - ($allavgmisses + $allavgmismatch + $allavgfp);
  $a_accuracy = 1 - ($allavgmisses + $allavgfp);
  if ($A_MOTA eq "yes") {
      # MOTP | misses| f.p. | mismatches | failed Corr | A-MOTA
      printf("RESULT: MOTP %3.0fmm\t miss %2.1f%%\t fp %2.1f%%\t failed Corr %2.1f%%\t A-MOTA %2.1f%%\n", 
	     $allavgdist, $allavgmisses*100, $allavgfp*100, $allavgfCorr*100, $a_accuracy*100 );
  }
  else {
      # MOTP | misses| f.p. | mismatches | MOTA
      printf("RESULT: MOTP %3.0fmm\t miss %2.1f%%\t fp %2.1f%%\t mismatches %d\t MOTA %2.1f%%\n", 
	     $allavgdist, $allavgmisses*100, $allavgfp*100, $allmismatch, $accuracy*100 );
  }
  printf("ABS TOTALS: groundT %.0f\t corr %.0f\t failedCorr %.0f\t distance %.0f\t miss %.0f\t falseP %.0f\t mismatch %0.f\n", 
	     $allground, $allcorr, $allfCorr, $alldist, $allmisses, $allfp, $allmismatch);
}
