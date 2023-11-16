#! /bin/bash

# a test script prepares the data and launches a test learning process.
# it only handles jpg to png conversion (imageMagick required)
# usage: ./run database

if [ $# -ne 1 ]
then
	echo -e "usage:\n\t./run.sh database\n"
	echo -e "\tdatabase refers to a folder containing some color images"
	echo -e "\t(I'll do the required conversion for you) but not the other way around.\n"
	echo -e "\tJPG and PNG supported. ImageMagick required.\n"
	exit 1
fi

# convert jpg to png if necessary

for img in $(find $1 -type f -iname "*.jpg")
do
	name=$(echo $img | sed 's:\(.*\)\.jpg:\1:g')
	convert $name.jpg $name.png && rm $name.jpg
	echo $(find $1 -type f -iname "*.jpg" | wc -l) " jpgs remain"
done

# sometimes database has both grayscale and RGB images

find $1 -type f -iname "*.png" | xargs file | grep "grayscale" | sed "s/:.*//g" > grayscales
find $1 -type f -iname "*.png" | xargs file | grep "RGB" | sed "s/:.*//g" > colors

# create the training and validation image lists

cat grayscales | grep small > validImages
cat grayscales | grep -v small > trainImages
rm -f colors grayscales
 
# compile

make distclean
make -Bj WITH_BLAS=mkl WITH_LOG_INFO=1 train

# a test run 

OPTS="-f -n 1e8 -p 7 7 -b 2 -r 3 -d 3 -t trainImages -v validImages -o OUT"
./train $OPTS 2>&1 | tee LOG
