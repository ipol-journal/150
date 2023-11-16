/**
 * @author Nicolas Limare <nicolas.limare@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "build.hpp"
#include "check_gradient.hpp"

/* call info */
static int _argc;
static const char * _arg0;

/**
 * usage information
 */
static void print_usage(void) {
	fprintf(stderr, \
	        "syntax:\n"
	        "\t%s -h\n"
	        "\t%s -g\n"
	        "\t%s [-f] [-r RATIO] [-d DEPTH] [-p ROWSIZE COLSIZE] [-b BORDER] [-n NB] [-s SIGMA] -t LIST -v LIST -o OUTPUT [-R]\n"
	        "options:\n"
	        "\t-h print this help\n"
	        "\t-g gradient checking\n"
	        "\t-f training from scratch (default false)\n"
	        "\t-r size ratio between a hidden layer and the input (default 2)\n"
	        "\t-d number of hidden layers (default 2)\n"
	        "\t-p training patch size (default 6 6)\n"
	        "\t-b border cropped out from teaching signals (default 2)\n"
	        "\t-n number of training rounds (default 1e7)\n"
	        "\t-s gaussian noise level (default 10)\n"
	        "\t-t list of training images\n"
	        "\t-v list of validation images\n"
	        "\t-o output file\n"
	        "\t-R reference mode for debugging:\n"
	        "\t\tno pseudo-random number generator initialization\n"
	        "\t\trandom network initialization\n"
	        "\t\tRMSE compared at every round\n"
	        "\t\tno partial result saved\n"
	        "\t\tunformatted output on stdout\n"
	        , _arg0, _arg0, _arg0);
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

int main(int argc, char *const *argv) {
	// default parameter values
	patchConfig patchSize;
	patchSize.rows = 7;
	patchSize.cols = 7;

	bool firstRun = false;
	int nHiddenLayers = 3;
	int ratio = 3;
	int border = 2;
	float sigma = 10;
	const char * tPath = "";
	const char * vPath = "";
	unsigned int stoopt = 1e7; // nb rounds of stochastic training
	const char * outfile = "";

	// default values for extern global variables
	flag_ref_mode = false;

	// set global call info
	_argc = argc;
	_arg0 = argv[0];

	// override parameters from command-line
	for (int i = 1; i < argc; i++) {
		// -g : gradientChecking
		if (0 == strcmp(argv[i],"-g")) {
			bool test_gradient = gradientChecking();
			exit(test_gradient ? EXIT_SUCCESS : EXIT_FAILURE);
		}
		// -h : usage help
		else if (0 == strcmp(argv[i],"-h")) {
			print_usage();
			exit(EXIT_SUCCESS);
		}
		// -f : first run
		else if (0 == strcmp(argv[i],"-f")) {
			firstRun = true;
		}
		// -p : training patch size
		else if (0 == strcmp(argv[i],"-p")) {
			patchSize.rows = atoi(argv[shift(i)]);
			patchSize.cols = atoi(argv[shift(i)]);
		}
		// -d : number of hidden layers
		else if (0 == strcmp(argv[i],"-d")) {
			nHiddenLayers = atoi(argv[shift(i)]);
		}
		// -r : size ratio between a hidden layer and the input
		// all the hidden layers are of the same size
		else if (0 == strcmp(argv[i],"-r")) {
			ratio = atoi(argv[shift(i)]);
		}
		// -n : nb rounds
		else if (0 == strcmp(argv[i],"-n")) {
			// as opposed to atoi, atof interprets scientific notation correctly
			stoopt = (unsigned int) atof(argv[shift(i)]);
		}
		// -t : training list
		else if (0 == strcmp(argv[i],"-t")) {
			tPath = argv[shift(i)];
		}
		// -b : border
		else if (0 == strcmp(argv[i],"-b")) {
			border = atoi(argv[shift(i)]);
		}
		// -s : sigma
		else if (0 == strcmp(argv[i],"-s")) {
			sigma = atof(argv[shift(i)]);
		}
		// -v : validation list
		else if (0 == strcmp(argv[i],"-v")) {
			vPath = argv[shift(i)];
		}
		// -o : output file
		else if (0 == strcmp(argv[i],"-o")) {
			outfile = argv[shift(i)];
		}
		// -R : reference mode
		else if (0 == strcmp(argv[i],"-R")) {
			flag_ref_mode = true;
			flag_random_seed = false;
		} else
			ERROR("unknown parameter");
	}

	if (patchSize.rows - 2*border <= 0 || patchSize.cols - 2*border <= 0)
		ERROR("unrealistic border cropping");

	buildAE(patchSize, nHiddenLayers, border,
	        ratio, tPath, vPath,
	        stoopt, outfile, firstRun, sigma);

	exit(EXIT_SUCCESS);
}
