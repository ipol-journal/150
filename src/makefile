# Copyright 2014 Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
#                Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
#
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved. This file is offered as-is,
# without any warranty.

CXXSRC		= $(wildcard *.cpp)
CSRC		= $(wildcard *.c)
CXXHDR		= $(wildcard *.hpp)
CHDR		= $(wildcard *.h)

CXXOBJ		= $(CXXSRC:.cpp=.o)
COBJ		= $(CSRC:.c=.o)
BM3D_OBJ	= ./bm3d/bm3d.o ./bm3d/utilities.o ./bm3d/lib_transforms.o ./bm3d/mt19937ar.o

CPPFLAGS	=
COPT		= -O2 -march=native -ftree-vectorize
CDBG		= -g
CFLAGS		= $(COPT) $(CDBG) -fopenmp
CXXFLAGS	= $(COPT) $(CDBG) -fopenmp
LDFLAGS		= -lgomp

default: all

ifdef WITH_TIMING
CPPFLAGS	+= -DUSE_TIMING
LDFLAGS		+= -lrt
endif

ifdef WITH_LOG_INFO
CPPFLAGS	+= -DLOGLEVEL=2
endif
ifdef WITH_LOG_DEBUG
CPPFLAGS	+= -DLOGLEVEL=3
endif

include makefile.blas

LDFLAGS		+= -lpng -lm -lfftw3f

all: train denoise

# generic object compilation

%.o: %.c
	$(CC) -c $< $(CPPFLAGS) $(CFLAGS) -o $@

%.o: %.cpp
	$(CHECK_MKLROOT)
	$(CXX) -c $< $(CPPFLAGS) $(CXXFLAGS) -o $@


# executable targets

train: 	Autoencoder.o NeuralStruct.o imageio.o build.o global.o \
		 check_gradient.o matops.o io_png.o timing.o train.o $(BM3D_OBJ)

	$(CXX) $^ $(LDFLAGS) -o $@

denoise: Autoencoder.o NeuralStruct.o imageio.o build.o global.o \
		matops.o io_png.o timing.o denoise.o $(BM3D_OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@

# doc

%.pdf: %.txt
	pandoc --from markdown \
		--standalone --normalize --smart --parse-raw \
		$< -o $@

# special targets

.PHONY:	clean distclean eigen ipolbm3d
clean:
	$(RM) $(CXXOBJ) $(COBJ) core
distclean: clean
	$(RM) train denoise
	$(RM) test/timing test/timing.o test/tanh test/tanh.o test/prng test/prng.o test/tmp/out.txt
	$(RM) README.pdf
	$(RM) inputFeatures.png outputFeatures.png testin.png testout.png teaching.png noisy.png context.png
	$(RM) *.pprof *.perf *.bak *.aux *.log

# install Eigen library locally
EIGEN_DIR	= eigen3
EIGEN_TGZ	= eigen-3.2.10.tar.gz
EIGEN_URL	= https://ipolcore.ipol.im/static/jyotsna/eigen-3.2.10.tar.gz

eigen: $(EIGEN_DIR)
$(EIGEN_TGZ):
	wget -q -O $@ $(EIGEN_URL)

$(EIGEN_DIR):	$(EIGEN_TGZ)
	$(RM) -r $@
	mkdir $@
	tar xfz $< -C $@ --strip-components=1

bm3d:
	@if [ ! -d bm3d ]; then \
	wget -q -O bm3d_src.tar.gz http://www.ipol.im/pub/art/2012/l-bm3d/bm3d_src.tar.gz; \
	tar xfz bm3d_src.tar.gz ; \
	mv bm3d_src bm3d; \
	rm -rf bm3d_src bm3d_src.tar.gz; fi

# code dependencies

makefile.dep: $(CSRC) $(CXXSRC) $(CHDR) $(CXXHDR)
	touch $@
	makedepend -Y -f $@ $(sort $(CSRC) $(CXXSRC)) 2>/dev/null

-include makefile.dep
