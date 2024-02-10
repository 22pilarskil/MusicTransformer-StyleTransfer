length=$1
length2=$2

sh generate.sh jazz 0 classical 1 $length
sh generate.sh jazz 0 classical 0 $length
sh generate.sh jazz 0 jazz 0 $length
sh generate.sh jazz 0 jazz 1 $length

#sh generate.sh classical 0 classical 1 $length
#sh generate.sh classical 0 classical 0 $length
#sh generate.sh classical 0 jazz 0 $length
#sh generate.sh classical 0 jazz 1 $length

#sh generate.sh jazz 0 classical 1 $length2
#sh generate.sh jazz 0 classical 0 $length2
#sh generate.sh jazz 0 jazz 0 $length2
#sh generate.sh jazz 0 jazz 1 $length2

#sh generate.sh classical 0 classical 1 $length2
#sh generate.sh classical 0 classical 0 $length2
#sh generate.sh classical 0 jazz 0 $length2
#sh generate.sh classical 0 jazz 1 $length2
