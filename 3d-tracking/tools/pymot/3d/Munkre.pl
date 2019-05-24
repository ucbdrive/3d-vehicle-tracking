#print "Included Munkre's algorithm!\n";
print "";

##
# Munkre's algorithm. 
# Input is the (n x m) cost function for assigning n objects to m hypotheses
# Output is an (n x m) matrix OutM with OutM(i,j) = 1 if i and j are assigned to each other, 0 otherwise.
#

sub munkre {
    my $HIGHEST_FLOAT = 5.0*(10**17);

    # C: 2-Dim (!) array(1..n,1..m) of float. This is the cost function for the assignment problem
    my @C = @_;
    my $n = @C;
    my $m = @{$C[0]};

    # Make sure n > m. If rotation necessary, set rot = 1
    my $rot = 0;
    if ($n < $m) {
	$rot = 1;
	$padding = $m-$n;
	my @TempC = ();
	for( my $j=0; $j<$m; $j++ ) {
	    push(@TempC, []);
	    for( my $i=0; $i<$n; $i++ ){ 
		push(@{$TempC[$j]}, $C[$i][$j]);
	    }
	}
	@C = @TempC;
	($n, $m) = ($m, $n);
    }

    # Funny! Looks like I have to make sure n=m anyway!?!
    my $padding = 0;
    if ($n != $m) {
	$padding = $n-$m;
	my @TempC = ();
	for( my $i=0; $i<$n; $i++ ) {
	    push(@TempC, []);
	    for( my $j=0; $j<$m; $j++ ){ 
		push(@{$TempC[$i]}, $C[$i][$j]);
	    }
	    for( my $j=0; $j<$padding; $j++ ){ 
		push(@{$TempC[$i]}, 5.0*10**16);
	    }
	}
	@C = @TempC;
	$m = $n;
    }

    # M: array(1..n,1..m) of integer. This is used to store stars and primes. For M[i][j]: 0=nothing, 1=star, 2=prime;
    my @M = ();
    for( my $i=0; $i<$n; $i++ ) {
	push(@M, []);
	for( my $j=0; $j<$m; $j++ ){
	    push(@{$M[$i]}, 0);
	} 
    }

    # Row,Col: array(1..n/m) of integer. This is to cover/uncover rows and columns. 0 = uncovered, 1 = covered
    my @R_cov = ();
    for( my $i=0; $i<$n; $i++ ) {
	push(@R_cov, 0);
    }
    my @C_cov = ();
    for( my $j=0; $j<$m; $j++ ) {
	push(@C_cov, 0);
    }

    # step: integer. Used to steer algorithm flow 
    my $step = 1;

    # done: boolean. End criterium reached? 
    my $done = 0;

    # Z0_r and Z0_c. These are used from step 4 to step 5 to represent the incovered prime zero
    my $Z0_r;
    my $Z0_c;
    




    # Step 1: For each row of the matrix, find the smallest element and subtract
    # it from every element in its row.  Go to Step 2.
    local *step1 = sub {
	my $minval;
	for (my $i=0; $i<$n; $i++) {
	    
	    $minval = $C[$i][0];
	    for (my $j=1; $j<$m; $j++){
		if ($minval>$C[$i][$j]) { 
		    $minval=$C[$i][$j];
		}
	    }
	
	    for (my $j=0; $j<$m; $j++){
		$C[$i][$j]=$C[$i][$j]-$minval; 
	    }

	} 
	
	$step=2; 
    };
    
    # Step 2: Find a zero (Z) in the resulting matrix.  If there is no starred
    # zero in its row or column, star Z. Repeat for each element in the matrix. Go to Step 3. 
    local *step2 = sub {
	for (my $i=0; $i<$n; $i++) {
	    for (my $j=0; $j<$m; $j++) {
		if ($C[$i][$j]==0 && $C_cov[$j]==0 && $R_cov[$i]==0) {
		    $M[$i][$j]=1; 
		    $C_cov[$j]=1; 
		    $R_cov[$i]=1; 
		}
	    }
	}

	for (my $j=0; $j<$m; $j++) {
	    $C_cov[$j]=0;
	} 
	for (my $i=0; $i<$n; $i++) {
	    $R_cov[$i]=0; 
	}
	
	$step=3; 
    };
    
    # Step 3: Cover each column containing a starred zero.  If K columns are covered,
    # the starred zeros describe a complete set of unique assignments.
    # In this case, Go to DONE, otherwise, Go to Step 4.
    local *step3 = sub {
	my $count;
	for (my $i=0; $i<$n; $i++) {
	    for (my $j=0; $j<$m; $j++) {
		if ($M[$i][$j]==1) {
		    $C_cov[$j]=1; 
		}
	    }
	}
	$count=0; 
	for (my $j=0; $j<$m; $j++) {
	    $count = $count + $C_cov[$j]; 
	}
	if ($count>=$m) {
	    $step=7; 
	} else {
	    $step=4; 
	}
    };

    # Step 4: Find a noncovered zero and prime it.  If there is no starred zero
    # in the row containing this primed zero, Go to Step 5.  Otherwise, cover this
    # row and uncover the column containing the starred zero. Continue in this manner
    # until there are no uncovered zeros left. Save the smallest uncovered value and Go to Step 6.
    local *step4 = sub {
	my $row;
	my $col;
	my $done;

	local *find_a_zero = sub {
	    my $i = 0;
	    my $j;
	    my $done = 0;

	    my $row = -1;
	    my $col = -1;
	    while(1==1) {
		$j=0; 
		while(1==1) {
		    if ($C[$i][$j]==0 && $R_cov[$i]==0 && $C_cov[$j]==0) { 
			$row=$i; 
			$col=$j; 
			$done=1; 
		    }
		    $j=$j+1; 
		    if ($j>=$m) {last;}
		}
		$i=$i+1; 
		if ($i>=$n) {$done=1;} 
		if ($done==1) {last;}
	    }
	    return ($row, $col);
	};


	local *star_in_row = sub {
	    my ($row) = @_;
	    my $tbool=0; 
	    for (my $j=0; $j<$m; $j++) {
		if ($M[$row][$j]==1){
		    $tbool=1; 
		}
	    }
	    return $tbool; 
	};


	local *find_star_in_row = sub {
	    my ($row) = @_;
	    my $col=-1; 
	    for (my $j=0; $j<$m; $j++) {
		if ($M[$row][$j]==1) {
		    $col=$j; 
		}
	    }
	    return $col;
	};

	$done=0; 
	while ($done==0) {
	    ($row, $col) = find_a_zero();
	    #print "Found a zero at ($row, $col)\n";
	    if ($row==-1) {
		$done=1; 
		$step=6;
	    } else {
		$M[$row][$col]=2;
		if (star_in_row($row)==1){ 
		    $col = find_star_in_row($row); 
		    $R_cov[$row]=1; 
		    $C_cov[$col]=0; 
		} else {
		    $done=1; 
		    $step=5; 
		    $Z0_r=$row; 
		    $Z0_c=$col; 
		}
	    }
	}
    };
    
    # Step 5: Construct a series of alternating primed and starred zeros as follows.
    # Let Z0 represent the uncovered primed zero found in Step 4.  Let Z1 denote the
    # starred zero in the column of Z0 (if any). Let Z2 denote the primed zero in the
    # row of Z1 (there will always be one).  Continue until the series terminates at a
    # primed zero that has no starred zero in its column.  Unstar each starred zero of
    # the series, star each primed zero of the series, erase all primes and uncover every
    #line in the matrix.  Return to Step 3.
    local *step5 = sub {
	my $count;
	my $done;
	my $r;
	my $c;
	my @path = ();

	local *find_star_in_col = sub {
	    my ($c) = @_;
	    my $r = -1; 
	    for (my $i=0; $i<$n; $i++) {
		if ($M[$i][$c]==1) {
		  $r=$i; 
		}
	    }
	    return $r;
	};


	local *find_prime_in_row = sub {
	    my ($r) = @_;
	    my $c = -1;
	    for (my $j=0; $j<$m; $j++) {
		if ($M[$r][$j]==2) {
		    $c=$j; 
		}
	    }
	    return $c;
	};


	local *convert_path = sub {
	    for (my $i=0; $i <= $count; $i++) {
		if ($M[$path[$i][0]][$path[$i][1]]==1) {
		    $M[$path[$i][0]][$path[$i][1]]=0; 
		} else {
		    $M[$path[$i][0]][$path[$i][1]]=1;
		} 
	    }
	};


	local *clear_covers = sub {
	    for (my $i=0; $i<$n; $i++) {
		$R_cov[$i]=0; 
	    }
	    for (my $j=0; $j<$m; $j++) {
		$C_cov[$j]=0;
	    } 
	};
	

	local *erase_primes = sub {
	    for (my $i=0; $i<$n; $i++) {
		for (my $j=0; $j<$m; $j++) {
		    if ($M[$i][$j]==2) {
			$M[$i][$j]=0; 
		    }
		}
	    }
	};


	$count=0; 
	push @path, [$Z0_r, $Z0_c]; 
	$done=0; 
	while ($done==0){
	    $c = $path[$count][1];
	    $r = find_star_in_col($c); 
	    if ($r!=-1) {
		$count=$count+1;
		push @path, [$r, $c];
	    } else { 
		$done=1;
	    } 
	    if ($done==0){
		$r = $path[$count][0];
		$c = find_prime_in_row($r); 
		$count=$count+1;
		push @path, [$r, $c];
	    }
	}
	convert_path();
	clear_covers();
	erase_primes();
      
	$step=3; 
    };

    # Step 6: Add the value found in Step 4 to every element of each covered row,
    # and subtract it from every element of each uncovered column.
    # Return to Step 4 without altering any stars, primes, or covered lines.
    local *step6 = sub {
	my $minval;

	local *find_smallest = sub {
	    my $minval = $HIGHEST_FLOAT;
	    for (my $i=0; $i<$n; $i++) {
		for (my $j=0; $j<$m; $j++) {
		    if ($R_cov[$i]==0 && $C_cov[$j]==0) { 
			if ($minval>$C[$i][$j]) {
			    $minval=$C[$i][$j]; 
			} 
		    }
		}
	    }
	    return $minval;
	};


	$minval = find_smallest(); 
	for (my $i=0; $i<$n; $i++) {
	    for (my $j=0; $j<$m; $j++) {
		if ($R_cov[$i]==1) {
		    $C[$i][$j]=$C[$i][$j]+$minval; 
		}
		if ($C_cov[$j]==0) {
		    $C[$i][$j]=$C[$i][$j]-$minval; 
		}
	    }
	}

	$step=4; 
    };


    # Main loop
    # if step is not between 1 and 6, we are done

#    my $limit = 0;
#    while ($done == 0 && $limit < 20){ 
#	
#	for (my $i=0; $i<$n; $i++) {
#	    for (my $j=0; $j<$m; $j++) {
#		print " $C[$i][$j]";
#	    }
#	    print "     ";
#	    for (my $j=0; $j<$m; $j++) {
#		if ($R_cov[$i] == 0 && $C_cov[$j] == 0) {
#		    print " O";
#		} else {
#		    print " X";
#		}
#	    }
#	    print "     ";
#	    for (my $j=0; $j<$m; $j++) {
#		print " $M[$i][$j]";
#	    }
#	    print "\n";
#	}
#
#	my $l = $limit+2;
#	print "$l. ";
#
#      CASE: 
#	{
#	    $step==1 && do {print "Step 1\n"; step1(); last CASE;};
#	    $step==2 && do {print "Step 2\n"; step2(); last CASE;};
#	    $step==3 && do {print "Step 3\n"; step3(); last CASE;};
#	    $step==4 && do {print "Step 4\n"; step4(); last CASE;};
#	    $step==5 && do {print "Step 5\n"; step5(); last CASE;};
#	    $step==6 && do {print "Step 6\n"; step6(); last CASE;};
#	    do {$done = 1; last CASE;};
#	}
#	$limit++;
#    }

    while ($done == 0) {
      CASE: 
	{
	    $step==1 && do {step1(); last CASE;};
	    $step==2 && do {step2(); last CASE;};
	    $step==3 && do {step3(); last CASE;};
	    $step==4 && do {step4(); last CASE;};
	    $step==5 && do {step5(); last CASE;};
	    $step==6 && do {step6(); last CASE;};
	    do {$done = 1; last CASE;};
	}
    }

    my @TempM = ();
    # Take away the padding
    if ($padding != 0) {
	$m = $n-$padding;
	for( my $i=0; $i<$n; $i++ ) {
	    push(@TempM, []);
	    for( my $j=0; $j<$m; $j++ ){ 
		push(@{$TempM[$i]}, $M[$i][$j]);
	    }
	}
    } else {
	@TempM = @M;
    }

    my @OutM = ();
    # If rotation was made, revert
    if ($rot==1) {
	for( my $j=0; $j<$m; $j++ ) {
	    push(@OutM, []);
	    for( my $i=0; $i<$n; $i++ ){ 
		push(@{$OutM[$j]}, $TempM[$i][$j]);
	    }
	}
    } else {
	@OutM = @TempM;
    }
    return @OutM;
}
