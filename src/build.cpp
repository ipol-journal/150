/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <utility>
#include <algorithm>

#include <omp.h>

#include "imageio.hpp"
#include "Autoencoder.hpp"
#include "NeuralStruct.hpp"

#include "matops.hpp"

#include "build.hpp"    // self-consistency

using namespace std;
using namespace Eigen;

#define TSIZE 1e3
#define VSIZE 1e4
#define INTERVAL 1e4

/**
 * print into ASCII a single precision float value
 * with just enough digits to guarantee that we get the value right
 * when read again with scanf(). This requires 8 or 16 digits digits
 *
 * For details, see
 *   http://randomascii.wordpress.com/2012/03/08/float-precisionfrom-zero-to-100-digits-2/
 */
inline _UNUSED
int fprintf_accurate(FILE * stream, const float value) {
	return fprintf(stream, "%+1.8e", value);
}

static
void write2disk(
        const char * file,
        VectorXf & obj,
        const char * objname,
        patchConfig const & patchSize,
        int hiddenSize,
        int nHiddenLayers,
        int border,
        float sigma
) {
	INFO("writing output to %s", file);

	// reference debug mode: dump the vector
	if (flag_ref_mode) {
		FILE * outfile;
		if (NULL == (outfile = fopen(file,"w")))
			ERROR("failed to open file %s", file);
		for (int i = 0; i < obj.size(); i++) {
			fprintf_accurate(outfile, obj[i]);
			fprintf(outfile, "\n");
		}
		fclose(outfile);
		return;
	}

	ofstream output;
	output.open(file, ios_base::trunc);
	output << "#include \"NeuralStruct.hpp\"" << endl;
	output << "float NeuralStruct::sigma = " << sigma << ";" << endl;
	output << "int NeuralStruct::patchSizeRows = " << patchSize.rows << ";" << endl;
	output << "int NeuralStruct::patchSizeCols = " << patchSize.cols << ";" << endl;
	output << "int NeuralStruct::hiddenSize = " << hiddenSize << ";" << endl;
	output << "int NeuralStruct::nHiddenLayers = " << nHiddenLayers << ";" << endl;
	output << "int NeuralStruct::border = " << border << ";" << endl;
	output.close();

	output.open(file, ios_base::app);

	//record one more attribute from optStackTheta
	if (string(objname).compare(string("optStackTheta")) == 0)
		output << "int NeuralStruct::thetaSize = " << obj.size() << ";" << endl;

	output << "float NeuralStruct::"<< objname << "[]={";
	output.precision(8);
	output.setf(ios::fixed, ios::floatfield);
	for (int i = 0; i < obj.size() - 1; i++)
		output << obj[i] << ",";
	output << obj[obj.size() - 1] << "};" << endl;
	output.close();
}

void dataProvider(
        MatrixXf & data_invisible,
        MatrixXf & data_visible,
        patchConfig const & patchSize,
        int border,
        vector<MatrixXf *> & Images,
        const char * Path,
        int SIZE,
        float sigma
) {
	// implement the noise adding routine
	drawPatches(data_invisible, Path, SIZE, patchSize, Images, 1);
	data_visible = data_invisible;

	maddg(data_visible.data(), data_visible.size(), sigma);
	MatrixXf data_invisible_copy = data_invisible;
	crop(data_invisible_copy, data_invisible, border, patchSize, 1);

	// scale the data
	whiten(data_visible.data(), data_visible.size());
	whiten(data_invisible.data(), data_invisible.size());
}

/**
 * wrapper of the NN learning procedures, including
 * supervised data drawing and learning progress monitoring
 */
#pragma GCC diagnostic ignored "-Wfloat-equal"
void buildAE(
        patchConfig const & patchSize,
        int const & nHiddenLayers,
        int const & border,
        int const & ratio,
        const char * tPath,  	//training
        const char * vPath,  	//validation
        unsigned int stoopt,    //number of rounds of stochastic training
        const char * outfile,   //output file
        bool const & firstRun,
        float sigma
) {
	INFO("%i thread(s), with OpenMP", omp_get_max_threads());

	int oSize = (patchSize.rows-2*border)*(patchSize.cols-2*border);
	int iSize = patchSize.rows * patchSize.cols;
	int hiddenSize = iSize*ratio;

	//set up a neural network
	trainParams inParams;
	inParams.layerSizes.resize(nHiddenLayers+2);
	inParams.layerSizes[0] = iSize;
	inParams.layerSizes[nHiddenLayers+1] = oSize;
	for (int l = 1; l < nHiddenLayers + 1; l++)
		inParams.layerSizes[l] = hiddenSize;
	inParams.linear = true;
	Autoencoder encoder(inParams, NULL, NULL);

	// initialize with a previous NN computation
	// if not in reference mode
	if (!flag_ref_mode && !firstRun) {
		if (patchSize.rows != NeuralStruct::getPatchSize(0)
		                || patchSize.cols != NeuralStruct::getPatchSize(1)
		                || border != NeuralStruct::getBorder()
		                || sigma != NeuralStruct::getSigma()
		                || nHiddenLayers != NeuralStruct::getnHiddenLayers()
		                || hiddenSize != NeuralStruct::getHiddenSize())
			ERROR("cannot continue training. It's a different NN.");
		float * theta = NeuralStruct::getTheta();
		encoder.stackTheta = Map<VectorXf>(theta, NeuralStruct::getThetaSize(), 1);
	}

	//prepare validation dataset
	vector<MatrixXf *> vImages, tImages;
	MatrixXf vdata_invisible, vdata_visible;

	dataProvider(vdata_invisible, vdata_visible, patchSize, border, vImages, vPath, VSIZE, sigma);

	//get rid of the validation image set
	int vnImages = vImages.size();
	for (int k = 0; k < vnImages; k++)
		delete [] vImages[k];


	float bestRMSE = encoder.calcRMSE(vdata_invisible, vdata_visible);
	INFO("Initial validation RMSE at %f", bestRMSE);
	VectorXf bestTheta = encoder.stackTheta;

	//backprop
	INFO("start %i rounds of stochastic training", stoopt);
	// nb rounds to compare RMSE: never on debug mode
	unsigned int rmse_cmp_freq = (flag_ref_mode ? 0 : INTERVAL);
	// nb rounds to update output: never on debug mode
	unsigned int output_write_freq = (flag_ref_mode ? 0 : 5 * rmse_cmp_freq);
	for (unsigned int r = 0; r < stoopt; r++) {
		TIMING_RESET(TIMER_LOOP);
		TIMING_RESET(TIMER_RAND);
		TIMING_RESET(TIMER_AXPB);
		TIMING_RESET(TIMER_TANH);
		TIMING_RESET(TIMER_MMPRC);
		TIMING_RESET(TIMER_MMT);
		TIMING_RESET(TIMER_MTMA);
		TIMING_RESET(TIMER_OMSQ);
		TIMING_RESET(TIMER_SUM);
		TIMING_RESET(TIMER_PATCH);
		TIMING_RESET(TIMER_GEMM);
		TIMING_RESET(TIMER_CROP);
		TIMING_TOGGLE(TIMER_LOOP);

		// build a list of random patches stored as columns in
		// tdata_invisible
		MatrixXf tdata_invisible, tdata_visible;
		dataProvider(tdata_invisible, tdata_visible, patchSize, border, tImages, tPath, TSIZE, sigma);

		// link these patches into en encoder class
		encoder.inputs = &tdata_visible;
		encoder.outputs = &tdata_invisible;


		// one step of gradient descent
		encoder.train();

		TIMING_TOGGLE(TIMER_LOOP);

		if (rmse_cmp_freq && (0 == (r % rmse_cmp_freq))) {
			float rmse = encoder.calcRMSE(vdata_invisible, vdata_visible);
			INFO("round %i : validation RMSE at %f", r, rmse);
			if (rmse < bestRMSE) {
				INFO("best ever RMSE so far is %f", rmse);
				bestRMSE = rmse;
				bestTheta = encoder.stackTheta;
			}
		}
		//precaution in case of accidental failure
		if (output_write_freq && (0 == (r % output_write_freq)))
			write2disk(outfile, bestTheta,
			           "optStackTheta", patchSize, hiddenSize,
			           nHiddenLayers, border, sigma);

		TIMING_LOG("mmprc", TIMER_MMPRC);
		TIMING_LOG("mmT  ", TIMER_MMT);
		TIMING_LOG("mTma ", TIMER_MTMA);
		TIMING_LOG("tanh ", TIMER_TANH);
		TIMING_LOG("sum  ", TIMER_SUM);
		TIMING_LOG("omsq ", TIMER_OMSQ);
		TIMING_LOG("rand ", TIMER_RAND);
		TIMING_LOG("patch", TIMER_PATCH);
		TIMING_LOG("axpb ", TIMER_AXPB);
		TIMING_LOG("crop ", TIMER_CROP);
		TIMING_LOG("loop ", TIMER_LOOP);
		TIMING_LOG("gemm ", TIMER_GEMM);
		TIMING_PRINTF("\n");
	}

	//do this to get rid of the training image set
	int tnImages = tImages.size();
	for (int k = 0; k < tnImages;  k++)
		delete [] tImages[k];

	//record
	if (!rmse_cmp_freq)
		bestTheta = encoder.stackTheta;
	write2disk(outfile, bestTheta, "optStackTheta", patchSize, hiddenSize, nHiddenLayers, border, sigma);
}
#pragma GCC diagnostic warning "-Wfloat-equal"

void onePass(
        MatrixXf * noised,
        Autoencoder & encoder,
        patchConfig const & patchSize,
        int border,
        MatrixXf *& denoised
) {
	MatrixXf * input = image2patches(noised, 1, patchSize, 1);
	whiten(input->data(), input->size());

	// neural network inference
	vector<MatrixXf> activations(encoder.nHiddenLayers+1);
	encoder.predict(input[0], activations);
	MatrixXf * A = &activations[encoder.nHiddenLayers]; // short alias
	blacken(A->data(), A->size());

	// aggregate to have the end result
	patchConfig outPatch;
	outPatch.rows = patchSize.rows - 2*border;
	outPatch.cols = patchSize.cols - 2*border;
	denoised = patches2image(A, noised->rows()-2*border, noised->cols()-2*border, 1, outPatch, 1);
	delete [] input;
}


void denoise(
        const char * clean,
        const char * bm3dd,
        const char * nnd,
        const char * ssannd,
        const char * noised
) {
	int rows, cols;
	// force conversion
	MatrixXf * cleanI = rgb2gray(clean, rows, cols);

	// keep the original version for comparison later on
	MatrixXf * original = new MatrixXf [1];
	original[0] = cleanI[0];

	// add noise and run bm3d
	maddg(cleanI->data(), cleanI->size(), NeuralStruct::getSigma());
	MatrixXf * bm3d = bm3df(NeuralStruct::getSigma(), cleanI);

	patchConfig patchSize;
	patchSize.rows = NeuralStruct::getPatchSize(0);
	patchSize.cols = NeuralStruct::getPatchSize(1);
	assert(patchSize.cols == patchSize.rows);

	// nnoisy for nn
	MatrixXf * nnoisy = extend(cleanI, patchSize.cols/2);
	imwrite(noised, cleanI, 1);

	// set up the trained neural network
	int border = NeuralStruct::getBorder();
	trainParams inParams;
	inParams.layerSizes.resize(NeuralStruct::getnHiddenLayers()+2);
	inParams.layerSizes[NeuralStruct::getnHiddenLayers()+1] = (patchSize.rows-2*border)*(patchSize.cols-2*border);
	inParams.layerSizes[0] = patchSize.rows*patchSize.cols;
	for (int l = 1; l < NeuralStruct::getnHiddenLayers()+1; l++)
		inParams.layerSizes[l] = NeuralStruct::getHiddenSize();
	inParams.linear = true;
	Autoencoder encoder(inParams, NULL, NULL);
	encoder.stackTheta = Map<VectorXf>(NeuralStruct::getTheta(), NeuralStruct::getThetaSize(), 1);

	// display extracted features
	vector<layerConfig> nnStruct(encoder.nHiddenLayers+1);
	encoder.theta2stack(nnStruct);
	display(nnStruct[0].W.transpose(), nnStruct[0].W.rows(), patchSize, "inputf.png");

	// patch denoise and border cropping
	MatrixXf * nndenoised = NULL;
	onePass(nnoisy, encoder, patchSize, border, nndenoised);

	int margin = (patchSize.rows-border*2)/2;
	MatrixXf nncropped = nndenoised->block(margin, margin, original->rows(), original->cols());
	nndenoised->resize(original->rows(), original->cols());
	nndenoised[0] = nncropped;

	// pixel selection based on a texture detector
	MatrixXi nfriends;
	int windowSize = 7;
	float threshold = NeuralStruct::getSigma() < 10.1 ? 1.7 : (NeuralStruct::getSigma() < 25.1 ? 1.2 : 1.1);
	count(nnoisy, windowSize, patchSize, threshold, NeuralStruct::getSigma(), nfriends);

	// and combine to form SSaNN
	MatrixXf * mark = new MatrixXf [1];
	MatrixXf * combined = new MatrixXf [1];
	mark->resize(nfriends.rows(), nfriends.cols());
	combined->resize(nfriends.rows(), nfriends.cols());
	float * mark_ptr = mark->data();
	float * com_ptr = combined->data();
	float * nn_ptr = nndenoised->data();
	float * bm3d_ptr = bm3d->data();
	int * nf_ptr = nfriends.data();
	for (int k = 0; k < mark->size(); k++) {
		mark_ptr[k] = nf_ptr[k] < 4 ? 0 : 250;
		com_ptr[k] = nf_ptr[k] < 4 ? nn_ptr[k] : bm3d_ptr[k];
	}

	delete [] nnoisy;

	float rmse = calcRMSE(original, nndenoised);
	INFO("neural network image RMSE of %s at %.02f", clean, rmse);
	rmse = calcRMSE(original, bm3d);
	INFO("BM3D image RMSE of %s at %.02f", clean, rmse);
	rmse = calcRMSE(original, combined);
	INFO("SSaNN image RMSE of %s at %.02f", clean, rmse);

	imwrite("clean.png", original, 1);
	imwrite(bm3dd, bm3d, 1);
	imwrite(nnd, nndenoised, 1);
	imwrite("mark.png", mark, 1);
	imwrite(ssannd, combined, 1);
}
