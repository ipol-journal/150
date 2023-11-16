/**
 * @author Yi-Qing Wang <yiqing.wang@cmla.ens-cachan.fr>
 * All rights reserved.
 */

#include "global.hpp"

#include <iostream>
#include <fstream>

#include <omp.h>

#include "matops.hpp"
#include "io_png.h"
#include "prng.hpp"	//external random number generator

#include "imageio.hpp" // self-consistency

#include "bm3d/bm3d.h"

#define OPP 2
#define DCT 4
#define BIOR 5

#define REDC 0.2989
#define GREC 0.5870
#define BLUC 0.1140

using namespace std;
using namespace Eigen;

MatrixXf * imread(
        const char * fileName,
        int & nRows,
        int & nCols,
        int & nChannels
) {
	size_t nx, ny, nc;
	float * pixelStream = read_png_f32(fileName, &nx, &ny, &nc);
	if (pixelStream == NULL)
		ERROR("Unable to get %s", fileName);

	//return these parameters
	nCols = (int)nx;
	nRows = (int)ny;
	nChannels = (int)nc;
	DEBUG("read in %s of dimension: %lu , %lu , %i",
	      fileName, ny, nx, nChannels);

	//input stream assumes row-major while Eigen defaults to column-major
	Map<MatrixXf> parallel(pixelStream, nCols, nRows*nChannels);
	MatrixXf * image = new MatrixXf [nChannels];
	for (int ch = 0; ch < nChannels; ch++)
		image[ch] = parallel.block(0, ch*nRows, nCols, nRows).transpose();

	//release
	free(pixelStream);

	return image;
}

void imwrite(
        const char * fileName,
        MatrixXf * image,
        int nChannels
) {
	//allocate
	int nCols = image->cols();
	int nRows = image->rows();
	int pixelsPerChannel = nCols*nRows;
	float * output = new float [pixelsPerChannel*nChannels];

	//this part should be straightforward but still be careful with the order
	for (int j = 0; j < nCols; j++)
		for (int i = 0; i < nRows; i++)
			for (int ch = 0; ch < nChannels; ch++)
				output[ch*pixelsPerChannel + i*nCols + j] = image[ch](i,j);

	//output image
	write_png_f32(fileName, output, (size_t) nCols, (size_t) nRows, (size_t) nChannels);
	INFO("write %s to the local folder", fileName);

	//release
	delete [] output;
	delete [] image;
}


void display(
        MatrixXf const & data,
        int nPieces,
        patchConfig const & patchSize,
        const char * pic
) {
	assert(nPieces <= data.cols());

	//output images always have 10 columns
	int dispNcols = 21;
	int dispNrows = nPieces % dispNcols == 0 ? nPieces / dispNcols : int(nPieces / dispNcols) + 1;

	int padding = 1;
	//allocate for output
	MatrixXf * output = new MatrixXf [3];
	for (int ch = 0; ch < 3; ch++)
		output[ch] = MatrixXf::Constant(dispNrows * (patchSize.rows + padding) + padding,
		                                dispNcols * (patchSize.cols + padding) + padding,
		                                255);

	//fill
	int nRows = data.rows();
	int unit = patchSize.rows * patchSize.cols;
	int isColor = nRows == unit ? 0 : 1;
	for (int p = 0; p < nPieces; p++) {
		VectorXf pvec = data.col(p).cast<float>();
		pvec -= VectorXf::Constant(nRows, pvec.minCoeff());
		pvec *= 255 / pvec.maxCoeff();
		int row = p / dispNcols;
		int col = p % dispNcols;
		for (int ch = 0; ch < 3; ch++) {
			VectorXf pvech = pvec.segment(ch * isColor * unit, unit);
			output[ch].block(padding + row * (padding + patchSize.rows),
			                 padding + col * (padding + patchSize.cols),
			                 patchSize.rows, patchSize.cols) = Map<MatrixXf>(pvech.data(), patchSize.rows, patchSize.cols);
		}
	}

	//out
	imwrite(pic, output, 3);
}


static int pathFile2charArray(
        const char * pathFile,
        char **& addr
) {
	//initialize the returned value
	int pathCount = 0;

	for (int loop = 0; loop < 2; loop++) {

		ifstream myfile(pathFile);

		if (!myfile.is_open())
			ERROR("Unable to open %s", pathFile);

		string line;

		//an offset to make the line count (total) right
		int total = -1;

		while (myfile.good()) {

			//count lines
			getline(myfile, line);
			total += 1;

			//only do this at the second loop
			if (loop == 1 && total < pathCount) {
				//convert string to char array
				addr[total] = new char [line.size()+1];
				addr[total][line.size()] = 0;
				memcpy(addr[total], line.c_str(), line.size());
			}
		}

		myfile.close();

		//first loop just count the number of lines in the file
		if (loop == 0) {

			pathCount = total;

			if (pathCount < 1)
				ERROR("Empty %s", pathFile);

			//allocate
			addr = new char * [pathCount];
		}
	}

	return pathCount;

}

static void readImagesFromPathFile(
        const char * pathFile,
        vector<MatrixXf *> & images,
        int expectedNChannels
) {
	char ** addr = NULL;
	int nImages = pathFile2charArray(pathFile, addr);
	images.resize(nImages);

	int nRows, nCols, nChannels;
	for (int im = 0; im < nImages; im++) {
		const char * file = addr[im];
		images[im] = imread(file, nRows, nCols, nChannels);
		if (nChannels != expectedNChannels) {
			const char * type = nChannels == 1 ? "grayscale" : "RGB";
			ERROR("Detect a %s %s", type, file);
		}
	}

	//release
	for (int im = 0; im < nImages; im++)
		delete [] addr[im];
	delete [] addr;
}

/**
 * draw patches at random
 */
static void gatherPatches(
        MatrixXf & patches,
        patchConfig const & patchSize,
        int nPatches,
        vector<MatrixXf *> & images,
        int nChannels
) {
	int unit = patchSize.rows * patchSize.cols;
	patches.resize(unit*nChannels, nPatches);

	static bool firstrun = true;
	// initialize random state(s)
	static prng_state_s * state_omp; //one per thread
	if (firstrun) {
		int maxt = omp_get_max_threads();
		state_omp = new prng_state_s[maxt+1];
		for (int t = 0; t < maxt+1; t++) {
			state_omp[t] = prng_new_state();
			if (flag_random_seed)
				prng_init_auto(state_omp[t]);
		}
		firstrun = false;
	}

	DEBUG("ready to draw %i random patches of size %i x %i from images",
	      nPatches, patchSize, patchSize);

	int nImages = images.size();

	#pragma omp parallel
	{
		// TODO : out of parallel block, use "private/shared"
		float * patches_ = patches.data();
		size_t rows = patches.rows();
		// get state
		int t = omp_get_thread_num();
		prng_state_s state = state_omp[t];
		#pragma omp for schedule(static)
		for (int p = 0; p < nPatches; p++) {
			// TODO: provide prng_lessthan(int) to avoid unneeded int->float->int
			int imi = prng_unif(state) * nImages;
			int sr = (images[imi]->rows() - patchSize.rows) * prng_unif(state);
			int sc = (images[imi]->cols() - patchSize.cols) * prng_unif(state);
			DEBUG("patch %i : %i, %i, %i", p, imi, sr, sc);
			for (int ch = 0; ch < nChannels; ch++) {
				MatrixXf patch = images[imi][ch].block(sr, sc, patchSize.rows, patchSize.cols);
				float * patch_ = patch.data();
				for (int i= 0; i < unit; i++)
					patches_[p * rows + ch * unit + i] = (float) patch_[i];
			}
		}
		// save state
		state_omp[t] = state;
	}
}

static int calcNumPatches(
        int nPixels,
        int patchSize,
        int step
) {
//	it holds that for some k, nPixels = patchSize + step * k + something
//	with something = 0 to k-1
	int something = (nPixels - patchSize) % step;
	int k = (nPixels - something - patchSize)/step;
	return k + 1;
}

MatrixXf * image2patches(
        MatrixXf * image,
        int stride,
        patchConfig const & patchSize,
        int nChannels
) {

	int nRows = image->rows();
	int nCols = image->cols();

	//how many patches to draw from this image
	int rows = calcNumPatches(nRows, patchSize.rows, stride);
	int cols = calcNumPatches(nCols, patchSize.cols, stride);

	//constraints on patch position
	int row = -1 * patchSize.rows;
	int col = -1 * patchSize.cols;
	int maxRow = nRows - patchSize.rows;
	int maxCol = nCols - patchSize.cols;

	//practical stuff
	MatrixXf * patches = new MatrixXf [1];
	int unit = patchSize.rows * patchSize.cols;
	patches[0] = MatrixXf::Zero(unit*nChannels, rows*cols);

	int counter = 0;
	for (int i = 0; i < rows; i++) {
		row = max(0, min(maxRow, row + stride));
		col = -1 * patchSize.cols;
		for (int j = 0; j < cols; j++) {
			col = max(0, min(maxCol, col + stride));
			MatrixXf patch = MatrixXf::Zero(patchSize.rows, patchSize.cols);
			VectorXf cPatch = VectorXf::Zero(unit*nChannels);
			for (int ch = 0; ch < nChannels; ch++) {
				patch = image[ch].block(row, col, patchSize.rows, patchSize.cols);
				cPatch.segment(unit*ch, unit) = Map<VectorXf>(patch.data(), unit, 1).cast<float>();
			}
			patches->col(counter++) = cPatch;
		}
	}

	return patches;
}

MatrixXf * patches2image(
        MatrixXf * patches,
        int nRows,
        int nCols,
        int stride,
        patchConfig const & patchSize,
        int nChannels
) {
	MatrixXf * image = new MatrixXf [nChannels];
	for (int ch = 0; ch < nChannels; ch++)
		image[ch] = MatrixXf::Zero(nRows, nCols);

	//count how many times each pixel has been covered by some patch
	MatrixXi mask = MatrixXi::Zero(nRows, nCols);

	int rows = calcNumPatches(nRows, patchSize.rows, stride);
	int cols = calcNumPatches(nCols, patchSize.cols, stride);

	//constraints on patch position
	int col, row = -1 * patchSize.rows;
	int maxRow = nRows - patchSize.rows;
	int maxCol = nCols - patchSize.cols;

	int counter = 0;
	int unit = patchSize.rows * patchSize.cols;
	for (int i = 0; i < rows; i++) {
		row = max(0, min(maxRow,  row + stride));
		col = -1 * patchSize.cols;
		for (int j = 0; j < cols; j++) {
			col = max(0, min(maxCol, col + stride));
			VectorXf cPatch = patches->col(counter++);
			for (int ch = 0; ch < nChannels; ch++) {
				VectorXf patch = cPatch.segment(unit*ch, unit).cast<float>();
				image[ch].block(row, col, patchSize.rows, patchSize.cols) += Map<MatrixXf>(patch.data(), patchSize.rows, patchSize.cols);
			}
			mask.block(row, col, patchSize.rows, patchSize.cols) += MatrixXi::Ones(patchSize.rows, patchSize.cols);
		}
	}

	for (int j = 0; j < nCols; j++)
		for (int i = 0; i < nRows; i++)
			for (int ch = 0; ch < nChannels; ch++)
				image[ch](i, j) = (mask(i, j) == 0
				                   ? 0
				                   : image[ch](i, j) / mask(i, j));

	return image;
}


void drawPatches(
        MatrixXf & patches,
        const char * path,
        int nPatches,
        patchConfig const & patchSize,
        vector<MatrixXf *> & images,
        int nChannels
) {
	TIMING_TOGGLE(TIMER_PATCH);
	//gather images if necessary
	if (images.size() == 0) {
		readImagesFromPathFile(path, images, nChannels);
		INFO("read in %lu images in all from %s", images.size(), path);
	}

	gatherPatches(patches, patchSize, nPatches, images, nChannels);
	TIMING_TOGGLE(TIMER_PATCH);
}


void crop(
        const MatrixXf & input,
        MatrixXf & output,
        int border,
        patchConfig const & patchSize,
        int nChannels
) {
	size_t nPatches = input.cols();
	output.resize((patchSize.rows-2*border)*(patchSize.cols-2*border)*nChannels, nPatches);
	const float * in_data = input.data();
	float * out_data = output.data();

	size_t offset = border * patchSize.rows + border;

	TIMING_TOGGLE(TIMER_CROP);
	if (border == 0)
		// no border: shortcut to a single copy
		memcpy(out_data, in_data, input.size()*sizeof(float));
	else
		// loop over patches and channels
		for (size_t p = 0; p < nPatches * nChannels; p++) {
			const float * in_ptr = in_data + p * patchSize.rows * patchSize.cols;
			float * out_ptr = out_data + p * (patchSize.rows - 2*border) * (patchSize.cols - 2*border);
			// loop over patch lines and copy
			for (size_t k = 0; k < patchSize.cols - 2*border; k++)
				memcpy(out_ptr + k * (patchSize.rows - 2*border),
				       in_ptr + k * patchSize.rows + offset,
				       (patchSize.rows - 2*border) * sizeof(float));
		}
	TIMING_TOGGLE(TIMER_CROP);
}


void count(
        MatrixXf * noisy,
        int windowSize,
        patchConfig const & patchSize,
        float threshold,
        float sigma,
        MatrixXi & nfriends
) {
	threshold *= sigma*sigma*patchSize.rows*patchSize.cols*2*threshold;
	MatrixXf * nPatches = image2patches(noisy, 1, patchSize, 1);
	nfriends.resize(noisy->rows()-patchSize.rows+1, noisy->cols()-patchSize.cols+1);

	int mRows = nfriends.rows();
	int mCols = nfriends.cols();

	#pragma omp parallel for schedule(static)
	for (int row = 0; row < mRows; row++)
		for (int col = 0; col < mCols; col++) {

			int minr = row - windowSize;
			minr = minr < 0 ? 0 : minr;
			int maxr = row + windowSize;
			maxr = maxr > mRows ? mRows : maxr;

			int minc = col - windowSize;
			minc = minc < 0 ? 0 : minc;
			int maxc = col + windowSize;
			maxc = maxc > mCols ? mCols : maxc;

			int sum = 0;
			VectorXf ref = nPatches->col(row*mCols+col);
			for (int r = minr; r < maxr; r++)
				for (int c = minc; c < maxc; c++) {
					VectorXf current = nPatches->col(r*mCols+c);
					if ((ref - current).squaredNorm() < threshold)
						sum += 1;
				}
			nfriends(row, col) = sum;
		}

	delete [] nPatches;
}

// a wrapper around IPOL's bm3d

MatrixXf * bm3df(
        float sigma,
        MatrixXf * noisy
) {
	int nChannels = 1;
	int width = noisy->cols();
	int height = noisy->rows();
	int nPixels = width*height*nChannels;

	vector<float> noisy_vec, basic_vec, denoised_vec;
	noisy_vec.resize(nPixels);
	basic_vec.resize(nPixels);
	denoised_vec.resize(nPixels);
	float * noisy_ptr = noisy->data();
	for (int k = 0; k < nPixels; k++)
		noisy_vec[k] = noisy_ptr[k];

	run_bm3d(sigma, noisy_vec, basic_vec, denoised_vec, height, width, nChannels, 0, 1, BIOR, DCT, OPP);

	MatrixXf * denoised = new MatrixXf [1];
	denoised->resize(height, width);
	float * ptr = denoised->data();
	for (int k = 0; k < nPixels; k++)
		ptr[k] = denoised_vec[k];

	return denoised;
}

MatrixXf * extend(
        MatrixXf * input,
        int margin
) {
	MatrixXf * output = new MatrixXf [1];
	output->resize(input->rows()+margin*2, input->cols()+margin*2);
	float * out_data = output->data();
	float * in_data = input->data();

	// copy main body
	for (int col = margin; col < margin + input->cols(); col++) {
		float * out_ptr = out_data + col*output->rows() + margin;
		float * in_ptr = in_data + (col-margin)*input->rows();
		memcpy(out_ptr, in_ptr, input->rows()*sizeof(float));
	}

	// copy two vertical margins
	for (int col = 0; col < margin; col++) {
		float * out_ptr = out_data + col*output->rows() + margin;
		float * in_ptr = out_data + (2*margin-col)*output->rows() + margin;
		memcpy(out_ptr, in_ptr, input->rows()*sizeof(float));
		out_ptr = out_data + (margin+input->cols()+col)*output->rows() + margin;
		in_ptr = out_data + (margin+input->cols()-col-2)*output->rows() + margin;
		memcpy(out_ptr, in_ptr, input->rows()*sizeof(float));
	}

	// copy two horizontal margins
	for (int row = 0; row < margin; row++) {
		float * out_ptr = out_data + row;
		float * in_ptr = out_data - row + 2 * margin;
		for (int k = 0; k < output->cols(); k++)
			out_ptr[k * output->rows()] = in_ptr[k * output->rows()];
		out_ptr = out_data + margin + input->rows() + row;
		in_ptr = out_data + margin + input->rows() - row - 2;
		for (int k = 0; k < output->cols(); k++)
			out_ptr[k * output->rows()] = in_ptr[k * output->rows()];
	}

	return output;
}

float calcRMSE(
        MatrixXf * ref,
        MatrixXf * other
) {
	float mse = 0;
	float * in = ref->data();
	float * de = other->data();
	for (int k = 0; k < ref->size(); k++) {
		de[k] = de[k] > 255 ? 255 : de[k];
		de[k] = de[k] < 0 ? 0 : de[k];
		float diff = de[k] - in[k];
		mse += diff * diff;
	}
	return sqrt(mse/ref->size());
}


// MATLAB style color conversion
MatrixXf * rgb2gray(
        const char *input,
        int & nRows,
        int & nCols
) {
	int nChannels;
	MatrixXf * rgb = imread(input, nRows, nCols, nChannels);
	if (nChannels == 3) {
		MatrixXf * gray = new MatrixXf [1];
		gray->resize(nRows, nCols);
		float * out = gray->data();
		float * R = rgb[0].data();
		float * G = rgb[1].data();
		float * B = rgb[2].data();
		for (int k = 0; k < gray->size(); k++)
			out[k] = R[k]*REDC + B[k]*BLUC + G[k]*GREC;
		delete [] rgb;
		return gray;
	} else {
		INFO("%s is a grayscale image already.", input);
		return rgb;
	}
}
