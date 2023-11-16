/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <stdlib.h>

#include "build.hpp"

#include "NeuralStruct.hpp"

/* call info */
static int _argc;
static const char * _arg0;

static void usage() {
	fprintf(stderr, \
	        "syntax:\n"
	        "	%s -h\n"
	        "	%s -c clean [-n noisy] [-d nn_output] [-b bm3d_output] [-s ssann_output]\n"
	        "options:\n"
	        "    	-h print this help\n"
	        "	-c path to a clean input image\n"
	        "	-n path to store the noisy image (default ./noisy.png)\n"
	        "	-b path to store BM3D denoised image (default ./bm3d.png)\n"
	        "	-s path to store SSaNN denoised image (default ./ssann.png)\n"
	        "	-d path to store NN denoised image (default ./nn.png)\n",
	        _arg0, _arg0);
	exit(EXIT_SUCCESS);
}

/**
 * increment argument index without going too far
 */
static int shift(int &i) {
	i++;
	if (i >= _argc)
		ERROR("missing parameter value");
	return i;
}

int main(int argc, char ** argv) {

	const char * clean = "unspecified";
	const char * nn = "./nn.png";
	const char * bm3d = "./bm3d.png";
	const char * ssann = "./ssann.png";
	const char * noisy = "./noisy.png";

	// set global call info
	_argc = argc;
	_arg0 = argv[0];

	for (int i = 1; i < argc; i++) {
		if (0 == strcmp(argv[i],"-h")) {
			usage();
		} else if (0 == strcmp(argv[i],"-c")) {
			clean = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-d")) {
			nn = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-s")) {
			ssann = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-b")) {
			bm3d = argv[shift(i)];
		} else if (0 == strcmp(argv[i],"-n")) {
			noisy = argv[shift(i)];
		} else
			ERROR("unknown parameter");

	}

	if (0 == strcmp(clean, "unspecified"))
		ERROR("Unspecified input (print the manual with %s -h)", _arg0);

	denoise(clean, bm3d, nn, ssann, noisy);
}
