#pragma once
#include <cv.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <cmath>

#include "graph.h"

#define NUM_CLUSTER 5
#define GAMMA 2
#define LAMBDA 100
#define PI 3.141592653

enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );

private:
	void GMMInit(cv::InputArray _img, cv::InputArray _mask,
		cv::InputOutputArray _bgdModel, cv::InputOutputArray _fgdModel,
		cv::InputOutputArray _bgdData, cv::InputOutputArray _fgdData);

	void GMMGamma(cv::InputArray _bgdModel, cv::InputArray _fgdModel, cv::InputArray _bgdData, cv::InputArray _fgdData,
		cv::InputOutputArray _bgdGamma, cv::InputOutputArray _fgdGamma, cv::InputOutputArray _bgdK, cv::InputOutputArray _fgdK);

	float Gauss(cv::InputArray _X, cv::InputArray _params);

	void GMMUpdate(cv::InputArray _img, cv::InputArray _mask,
		cv::InputArray _bgdGamma, cv::InputArray _fgdGamma,
		cv::InputArray _bgdK, cv::InputArray _fgdK,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		cv::InputArray _bgdData,cv::InputArray _fgdData);

	void GMMUpdateData(cv::InputArray _img, cv::InputArray _mask,
		cv::InputOutputArray _bgdData,cv::InputOutputArray _fgdData);

	void MinCut(cv::InputArray _img, cv::InputOutputArray _mask,
		cv::InputArray _bgdModel, cv::InputArray _fgdModel,
		const cv::Mat leftW, const cv::Mat upleftW, const cv::Mat upW, const cv::Mat uprightW);

	float UWeight(cv::InputArray _pixel, cv::InputArray _model);

	void VWeight(cv::Mat& img, 
		cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW,
		float beta, float gamma);

	float getBeta(const cv::Mat& img);

public:
	~GrabCut2D(void);
};