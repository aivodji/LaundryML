#!/usr/bin/perl

#transpose.pl transposes the set-family-file/01 matrix)/transaction-data
# from horizontal representation to virtical representation
# if a number x is wrtten in yth line in the input file, then y is written in 
# the xth line in the output file.
# Both input and output file are from/to standard input/output

$ARGC = @ARGV;
if ( $ARGC < 0 ){
  printf ("transpose.pl [separator] < input-file > output-file\n");
  exit (1);
}
$count = 0;
$m = 0;

$sep = " ";
if ( $ARGC > 0 ){ $sep = $ARGV[0]; } 

while (<STDIN>){
  chomp;
  @eles = split($sep, $_);
  $all = @eles;
  foreach $item( @eles ){
    push ( @{$t[$item]}, $count );
    if ( $item > $m ){ $m = $item; }
  }
  $count++;
}

for ( $i=0 ; $i<=$m ; $i++ ){ print "@{$t[$i]}\n";}


