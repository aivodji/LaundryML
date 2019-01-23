#!/usr/bin/perl

#appendnum.pl appends the column number (column ID) to each item in each line
# the item-separator can be given by the first parameter
# input and output are done from/to standard input/output

$ARGC = @ARGV;
if ( $ARGC < 0 ){
  printf ("appendnum.pl:  [separator] < input-file > output-file\n");
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
  $c = 0;
  foreach $item( @eles ) {
    if ( $item ne "" ){
      print "$item.$c" ;
      $c++;
      if ($c<$all){ print $sep; }
    }
  }
  print "\n";
}
