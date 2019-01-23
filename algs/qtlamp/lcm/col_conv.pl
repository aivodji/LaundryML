#!/bin/perl

#!/usr/bin/perl

# according to the rule file, convert each item of each line to 
#  another item
# each line of the rule file is a rule for one column
# the ith line is for ith column
# the first item of each line means the type of conversion, of each column.

# #: comment
# I: ignore the column; no item will be added
# S: string, the column is treated as a string
# P: the column is treated as a number. The following sequence of 
#    numbers means the boundaries, and when the number in a column exceeds
#    the i-th number, then item XX_i is put. Thus, if the sequence is 
#    1 5 8 11, and the number is 6, then the items put are XX_3 and XX_4,
#    where XX is the column ID
# N: same as P, but items are put when the number is smaller than the boundaries.
# B: P and N
# M: items are put when the number in a column is smaller than a number in 
#    the former half, or bigger than a number in the latter half


# the intervals. The numbers will be classified according to the number
# ex) x y z, mean intervals (-inf, x), [x, y), [y,z), [z,+inf)

# The separator for the rule-file is " ", but we can specify the separator
# of the number in the data file, by the second parameter


$ARGC = @ARGV;
$c=0;
if ( $ARGC < 1 ){
  printf ("col_conv.pl: rule-file [separator] < input-file > output-file\n");
  exit (1);
}
open ( RULEFILE, "<$ARGV[0]" );
$c++;
$sep = " ";
if ( $ARGC > 1 ){ $sep = $ARGV[$c]; } 

@table = <RULEFILE>;
$count = 0;
%numbers = ();
$clms=0;

foreach $trans( @table ){
  chomp ( $trans );
  @eles = split(" ", $trans);
  $type[$clms] = shift(@eles);
  if ( $type[$clms] ne "#" ){
    if ( $type[$clms] ne "S" && $type[$clms] ne "I" ){
      $c=0;
      foreach $cell(@eles){
        if ( $cell == 0 ){
          if (index ( $cell, "0") >= 0 ){ $bound[$clms] .= $cell." "; }
        } else { $bound[$clms] .= $cell." "; }
        $c++;
      }
      $bound_num[$clms]=$c;
    }
    $clms++;
  }
}

while (<STDIN>){
  chomp;
  @eles = split($sep, $_);
  $c=0;
  foreach $cell(@eles){
    if ( $c >= $clms ){
      print $cell." ";
    } elsif ( $type[$c] eq "S" ){
      print "($c)".$cell." ";
    } elsif ( $type[$c] ne "I" ){
      @bnd=split(" ", $bound[$c]);
      $cc=0;
      foreach $bbb(@bnd){
        if ( ($type[$c] eq "P" || $type[$c] eq "B") && ($cell > $bbb) ){
          print "($c)>$bbb ";
        }
        if ( ($type[$c] eq "N" || $type[$c] eq "B") && ($cell < $bbb) ){
          print "($c)<$bbb ";
        }
        if ( ($type[$c] eq "M") && ($cell < $bbb) && ($cc < ($bound_num[$c]+1)/2) ){
          print "($c)<$bbb ";
        }
        if ( ($type[$c] eq "M") && ($cell > $bbb) && ($cc >= ($bound_num[$c]-1)/2)){
          print "($c)>$bbb ";
        }
        $cc++;
      }
    }
    $c++;
  }
  print "\n";
}



