#ifndef IM_H
#define IM_H

#include "global.hpp"

#include <string>
#include <vector>
#include "eigen.hpp"

using namespace Eigen;
using namespace std;

struct patchConfig {
	unsigned int rows;
	unsigned int cols;
};

MatrixXf * imread(
        const char * fileName,
        int & nRows,
        int & nCols,
        int & nChannels
);

void imwrite(
        const char * fileName,
        MatrixXf * image,
        int nChannels
);

MatrixXf * image2patches(
        MatrixXf * image,
        int stride,
        patchConfig const & patchSize,
        int nChannels
);

MatrixXf * patches2image(
        MatrixXf * patches,
        int nRows,
        int nCols,
        int stride,
        patchConfig const & patchSize,
        int nChannels
);

void display(
        MatrixXf const & data,
        int nPieces,
        patchConfig const & patchSize,
        const char * pic
);

void drawPatches(
        MatrixXf & patches,
        const char * path,
        int nPatches,
        patchConfig const & patchSize,
        vector<MatrixXf *> & images,
        int nChannels
);

void crop(
        const MatrixXf & input,
        MatrixXf & output,
        int border,
        patchConfig const & patchSize,
        int nChannels
);

void count(
        MatrixXf * noisy,
        int windowSize,
        patchConfig const & patchSize,
        float threshold,
        float sigma,
        MatrixXi & nfriends
);

MatrixXf * bm3df(
        float sigma,
        MatrixXf * noisy
);

MatrixXf * extend(
        MatrixXf * input,
        int margin
);

float calcRMSE(
        MatrixXf * ref,
        MatrixXf * other
);

MatrixXf * rgb2gray(
        const char * input,
        int & rows,
        int & cols
);

#endif
