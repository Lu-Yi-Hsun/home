reset
set ylabel 'time(sec)'
set style fill solid
set xtic("                False sharing    sharing"0)
set title 'perfomance comparison'
set term png enhanced font 'Verdana,10'
set output 'runtime.png'

plot [:][0:0.08]'output.txt' using 1 with histogram title 'False sharing', \
'' using 2 with histogram title 'sharing'  , \
 