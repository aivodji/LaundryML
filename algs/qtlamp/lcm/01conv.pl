#!/usr/bin/perl

# this script covert a 01 matrix to a to the transaction database, or
# adjacency matrix, 
# each line is converted to the sequence of columns which has value 1.

$ARGC = @ARGV;
if ( $ARGC < 0 ){
  printf ("01conv.pl:  [separator] < input-file > output-file\n");
  exit (1);
}
$count = 0;
%numbers = ();

$sep = " ";
if ( $ARGC >= 1 ){ $sep = $ARGV[0]; } 

while (<STDIN>){
  chomp;
  @eles = split($sep, $_);
  $all = @eles;
  $c = 1;
  foreach $item( @eles ) {
    if ( $item ne "0" ){
      print "$c" ;
      if ($c<$all){ print $sep; }
    }
    $c++;
  }
  print "\n";
}


