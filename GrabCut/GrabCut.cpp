#include "GrabCut.h"

GrabCut2D::~GrabCut2D(void)
{
}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
	cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    std::cout<<"Execute GrabCut Function: Please finish the code here!"<<std::endl;
	std::cout<<"mode : "<<mode<<std::endl;
	std::cout<<"rect : "<<rect<<std::endl;

	cv::Mat img = _img.getMat();
    cv::Mat& mask = _mask.getMatRef();
    cv::Mat& bgdModel = _bgdModel.getMatRef();
    cv::Mat& fgdModel = _fgdModel.getMatRef();

	cv::Mat bgdData;
	cv::Mat fgdData;
	cv::Mat bgdGamma;
	cv::Mat fgdGamma;
	cv::Mat bgdK;
	cv::Mat fgdK;

	cv::Mat leftW, upleftW, upW, uprightW;
	VWeight(img, leftW, upleftW, upW, uprightW, 
		getBeta(img), GAMMA);

	//std::cout<<"leftW :"<<std::endl<<leftW.row(0)<<std::endl;

	GMMInit(img,mask,bgdModel,fgdModel,bgdData,fgdData);

	//std::cout<<"fgdData : "<<std::endl<<fgdData<<std::endl;
	//std::cout<<"bgdData : "<<std::endl<<bgdData<<std::endl;
	//std::cout<<"fgdModel :"<<std::endl<<fgdModel<<std::endl;
	//std::cout<<"bgdModel :"<<std::endl<<bgdModel<<std::endl;

	for(int iter=0;iter<(iterCount<5?5:iterCount);iter++){
		GMMGamma(bgdModel,fgdModel,bgdData,fgdData,bgdGamma,fgdGamma,bgdK,fgdK);

		//std::cout<<"fgdGamma : "<<std::endl<<fgdGamma<<std::endl;
		//std::cout<<"bgdGamma : "<<std::endl<<bgdGamma<<std::endl;
		//std::cout<<"fgdK : "<<std::endl<<fgdK<<std::endl;
		//std::cout<<"bgdK : "<<std::endl<<bgdK<<std::endl;

		GMMUpdate(img,mask,bgdGamma,fgdGamma,bgdK,fgdK,
			bgdModel,fgdModel,bgdData,fgdData);

		//std::cout<<"fgdModel :"<<fgdModel<<std::endl;
		//std::cout<<"bgdModel :"<<bgdModel<<std::endl;

		MinCut(img, mask, bgdModel, fgdModel, leftW, upleftW, upW, uprightW);

		GMMUpdateData(img, mask, bgdData, fgdData);
	}
	

//一.参数解释：
	//输入：
	 //cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
     //cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
  	//int iterCount :           :每次分割的迭代次数（类型-int)


	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)

//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
	
}

void GrabCut2D::MinCut(cv::InputArray _img, cv::InputOutputArray _mask,
	cv::InputArray _bgdModel, cv::InputArray _fgdModel,
	const cv::Mat leftW, const cv::Mat upleftW, const cv::Mat upW, const cv::Mat uprightW)
{
	cv::Mat img = _img.getMat();
    cv::Mat& mask = _mask.getMatRef();
	cv::Mat bgdModel = _bgdModel.getMat();
    cv::Mat fgdModel = _fgdModel.getMat();

	typedef Graph<float,float,float> GraphType;
	GraphType *g = new GraphType(
		img.rows * img.cols,
		2*(4*img.rows * img.cols - 3*(img.cols + img.rows) + 2)
	);

	int index=0;
	for(int row=0;row<img.rows;row++){
		for(int col=0;col<img.cols;col++){
			g->add_node();
			cv::Mat pixel;
			pixel.create(1,3,CV_32F);
			for( int z=0; z<3; z++){
				pixel.at<float>(0, z) = img.at<cv::Vec3b>(row,col)[z];
			}
			float fgdWeight=UWeight(pixel,fgdModel);
			float bgdWeight=UWeight(pixel,bgdModel);
			//std::cout<<"pixel :"<<pixel<<std::endl;
			//std::cout<<"fgdWeight :"<<fgdWeight<<std::endl;
			//std::cout<<"bgdWeight :"<<bgdWeight<<std::endl;
			if(mask.at<uchar>(row,col) == 2 || mask.at<uchar>(row,col) == 3)
				g->add_tweights(index, fgdWeight,bgdWeight);
			else if(mask.at<uchar>(row,col)==0)
				//g->add_tweights(index, 0, LAMBDA);
				g->add_tweights(index, 0, bgdWeight);
			else if(mask.at<uchar>(row,col)==1)
				//g->add_tweights(index, LAMBDA, 0);
				g->add_tweights(index, fgdWeight, 0);
			index++;
		}
	}
	index=0;
	for(int row=0;row<img.rows;row++){
		for(int col=0;col<img.cols;col++){
			if( col>0 ){
                float w = leftW.at<float>(row,col);  
                g->add_edge( index, index-1, w, w );  
            }
            if( col>0 && row>0 ){  
                float w = upleftW.at<float>(row,col);  
                g->add_edge( index, index-img.cols-1, w, w );  
            }
            if( row>0 ){
                float w = upW.at<float>(row,col);  
                g->add_edge( index, index-img.cols, w, w );  
            }
            if( col<img.cols-1 && row>0 ){
                float w = uprightW.at<float>(row,col);  
                g->add_edge( index, index-img.cols+1, w, w );  
            }
			index++;
		}
	}
	float flow=g->maxflow();
	std::cout<<"flow :"<<flow<<std::endl;

	index=0;
	for(int row=0;row<img.rows;row++){
		for(int col=0;col<img.cols;col++){
			if (g->what_segment(index) == GraphType::SOURCE){
				if(mask.at<uchar>(row,col) == 2 || mask.at<uchar>(row,col) == 3)
					mask.at<uchar>(row,col)=2;
			}else{
				if(mask.at<uchar>(row,col) == 2 || mask.at<uchar>(row,col) == 3)
					mask.at<uchar>(row,col)=3;
			}
			index++;
		}
	}
}

void GrabCut2D::VWeight( cv::Mat& img,
	cv::Mat& leftW, cv::Mat& upleftW, cv::Mat& upW, cv::Mat& uprightW,
	float beta, float gamma)
{
	leftW.create( img.rows, img.cols, CV_32F );  
    upleftW.create( img.rows, img.cols, CV_32F );  
    upW.create( img.rows, img.cols, CV_32F );  
    uprightW.create( img.rows, img.cols, CV_32F );  

	const float gammaSqrt = gamma / std::sqrt(2.0f);

	for( int row = 0; row < img.rows; row++ )  
    {  
        for( int col = 0; col < img.cols; col++ )  
        {  
            cv::Vec3d color = img.at<cv::Vec3b>(row,col);  
            if( col>0 ){
                cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row,col-1);  
                leftW.at<float>(row,col) = (float)(gamma * exp(-beta*diff.dot(diff)));  
            }
            else  
                leftW.at<float>(row,col) = 0;

            if( col>0 && row>0 ){  
                cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row-1,col-1);  
                upleftW.at<float>(row,col) = (float)(gammaSqrt * exp(-beta*diff.dot(diff)));  
            }  
            else
                upleftW.at<float>(row,col) = 0;  

            if( row>0 ) {
                cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row-1,col);  
                upW.at<float>(row,col) = (float)(gamma * exp(-beta*diff.dot(diff)));  
            }  
            else  
                upW.at<float>(row,col) = 0;  

            if( row>0&&col<img.cols-1 ){  
                cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row-1,col+1);  
                uprightW.at<float>(row,col) = (float)(gammaSqrt * exp(-beta*diff.dot(diff)));  
            }
            else  
                uprightW.at<float>(row,col) = 0;  
        }
    }
}

float GrabCut2D::getBeta(const cv::Mat& img)
{
	float beta=0;
	for(int row=0; row<img.rows; row++){
		for(int col=0; col<img.cols; col++){
			cv::Vec3d color=img.at<cv::Vec3b>(row,col);
			if(col>0){
				cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row,col-1);
				beta += (float)diff.dot(diff);
			}
			if(row>0&&col>0){
				cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row-1,col-1);
				beta += (float)diff.dot(diff);	
			}
			if(row>0){
				cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row-1,col);
				beta += (float)diff.dot(diff);
			}
			if(row>0&&col<img.cols-1){
				cv::Vec3d diff = color - (cv::Vec3d)img.at<cv::Vec3b>(row-1,col+1);
				beta += (float)diff.dot(diff);
			}
		}
	}
	beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );
	//std::cout<<"beta :"<<beta<<std::endl;
	return beta;
}

float GrabCut2D::UWeight(cv::InputArray _pixel, cv::InputArray _model)
{
	cv::Mat pixel=_pixel.getMat();
	cv::Mat model=_model.getMat();

	cv::Mat gamma;
	gamma.create(1,NUM_CLUSTER,CV_32F);

	for( int cluster=0;cluster<NUM_CLUSTER;cluster++){
		cv::Mat params=model.row(cluster);
		gamma.at<float>(0,cluster)=Gauss(pixel,params);
	}
	//std::cout<<"gammma :"<<gamma<<std::endl;
	float maxP=gamma.at<float>(0,0);
	int maxIndex=0;
	for (int i=1;i<NUM_CLUSTER;i++){
		if(gamma.at<float>(0,i)>maxP){
			maxP=gamma.at<float>(0,i);
			maxIndex=i;
		}
	}
	//std::cout<<"maxIndex :"<<maxIndex<<std::endl;
	
	cv::Mat params=model.row(maxIndex);
	cv::Mat mean = params(cv::Range::all(),cv::Range(0,3));
	cv::Mat corr;
	corr.create(3, 3, CV_32F);
	corr.at<float>(0,0)=params.at<float>(3);corr.at<float>(0,1)=params.at<float>(4);corr.at<float>(0,2)=params.at<float>(5);
	corr.at<float>(1,0)=params.at<float>(6);corr.at<float>(1,1)=params.at<float>(7);corr.at<float>(1,2)=params.at<float>(8);
	corr.at<float>(2,0)=params.at<float>(9);corr.at<float>(2,1)=params.at<float>(10);corr.at<float>(2,2)=params.at<float>(11);
	cv::Mat corrInv = corr.inv();
	float pi = params.at<float>(0,12);

	cv::Mat delta=pixel-mean;
	float distance=0;
	float tmpDistance[3]={0,0,0};
	for (int col=0;col<3;col++){
		for (int row=0;row<3;row++){
			tmpDistance[col]+=delta.at<float>(0,row)*corrInv.at<float>(row,col);
		}
	}
	for (int i=0;i<3;i++){
		distance+=tmpDistance[i]*delta.at<float>(0,i);
	}

	float weight=0;
	weight=-(float)std::log(pi)+(float)(1./2.*std::log(std::fabs(cv::determinant(corr))))+(float)(1./2.*distance);
	//std::cout<<"std::log(pi) :"<<std::log(pi)<<std::endl;
	//std::cout<<"1./2.*std::log(std::fabs(cv::determinant(corr))) :"<<1./2.*std::log(std::fabs(cv::determinant(corr)))<<std::endl;
	//std::cout<<"1./2.*distance :"<<1./2.*distance<<std::endl;

	return weight;
}

void GrabCut2D::GMMInit(cv::InputArray _img, cv::InputArray _mask,
	cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
	cv::InputOutputArray _bgdData,cv::InputOutputArray _fgdData)
{
	cv::Mat img = _img.getMat();
    cv::Mat mask = _mask.getMat();
    cv::Mat& bgdModel = _bgdModel.getMatRef();
    cv::Mat& fgdModel = _fgdModel.getMatRef();
	cv::Mat& bgdData = _bgdData.getMatRef();
	cv::Mat& fgdData = _fgdData.getMatRef();

	int numFgd=0;
	int numBgd=0;

	//std::cout<<mask.type()<<std::endl;

	for( int row=0; row<mask.rows; row++){
		for( int col=0; col<mask.cols; col++){
			if	(mask.at<uchar>(row,col)%2==1){
				numFgd++;
			}else{
				numBgd++;
			}
		}
	}

	fgdData.create(numFgd,3,CV_32F);
	bgdData.create(numBgd,3,CV_32F);

	numFgd=0;
	numBgd=0;
	for( int row=0; row<mask.rows; row++){
		for( int col=0; col<mask.cols; col++){
			if	(mask.at<uchar>(row,col)%2==1){
				for( int z=0; z<3; z++){
					fgdData.at<float>(numFgd, z) = img.at<cv::Vec3b>(row,col)[z];
				}
				numFgd++;
			}else{
				for( int z=0; z<3; z++){
					bgdData.at<float>(numBgd, z) = img.at<cv::Vec3b>(row,col)[z];
				}
				numBgd++;
			}
		}
	}

	//std::cout<<"fgdData :"<<fgdData.row(0)<<std::endl;
	//std::cout<<"bgdData :"<<bgdData<<std::endl;

	cv::Mat fgdLabels;
	cv::Mat bgdLabels;

	kmeans(fgdData, NUM_CLUSTER, fgdLabels,
		cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
        3, cv::KMEANS_PP_CENTERS);
	kmeans(bgdData, NUM_CLUSTER, bgdLabels,
		cv::TermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
        3, cv::KMEANS_PP_CENTERS);

	//std::cout<<"fgdLabels :"<<fgdLabels<<std::endl;
	//std::cout<<"bgdLabels :"<<bgdLabels<<std::endl;

	fgdModel.create(NUM_CLUSTER, 13, CV_32F);
	bgdModel.create(NUM_CLUSTER, 13, CV_32F);

	cv::Mat tmpMeans=cv::Mat::zeros(NUM_CLUSTER, 4, CV_32F);

	for( int row=0; row<fgdLabels.rows; row++){
		//std::cout<<row<<" : "<<fgdLabels.at<int>(row,1)<<std::endl;
		for (int i=0;i<3;i++){
			tmpMeans.at<float>(fgdLabels.at<int>(row,0),i)+=
				fgdData.at<float>(row,i);
		}
		tmpMeans.at<float>(fgdLabels.at<int>(row,0),3)+=1;
	}
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<3; j++){
			fgdModel.at<float>(i,j)=tmpMeans.at<float>(i,j)/tmpMeans.at<float>(i,3);
		}
		fgdModel.at<float>(i,12)=tmpMeans.at<float>(i,3)/numFgd;
	}

	tmpMeans=cv::Mat::zeros(NUM_CLUSTER, 4, CV_32F);

	for( int row=0; row<bgdLabels.rows; row++){
		//std::cout<<row<<" : "<<fgdLabels.at<int>(row,1)<<std::endl;
		for (int i=0;i<3;i++){
			tmpMeans.at<float>(bgdLabels.at<int>(row,0),i)+=
				bgdData.at<float>(row,i);
		}
		tmpMeans.at<float>(bgdLabels.at<int>(row,0),3)+=1;
	}
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<3; j++){
			bgdModel.at<float>(i,j)=tmpMeans.at<float>(i,j)/tmpMeans.at<float>(i,3);
		}
		bgdModel.at<float>(i,12)=tmpMeans.at<float>(i,3)/numBgd;
	}

	cv::Mat tmpCor=cv::Mat::zeros(NUM_CLUSTER, 10, CV_32F);

	for( int row=0; row<fgdLabels.rows; row++){
		for (int i=0;i<3;i++){
			for (int j=0; j<3; j++){
				tmpCor.at<float>(fgdLabels.at<int>(row,0),i*3+j)+=
					(fgdData.at<float>(row,i)-fgdModel.at<float>(fgdLabels.at<int>(row,0),i))*
					(fgdData.at<float>(row,j)-fgdModel.at<float>(fgdLabels.at<int>(row,0),j));
			}
		}
		tmpCor.at<float>(fgdLabels.at<int>(row,0),9)+=1;
	}
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<9; j++){
			fgdModel.at<float>(i,3+j)=tmpCor.at<float>(i,j)/tmpCor.at<float>(i,9);
		}
	}
	//std::cout<<"tmpCor :"<<std::endl<<tmpCor<<std::endl;

	tmpCor=cv::Mat::zeros(NUM_CLUSTER, 10, CV_32F);

	for( int row=0; row<bgdLabels.rows; row++){
		//std::cout<<row<<" : "<<fgdLabels.at<int>(row,1)<<std::endl;
		for (int i=0;i<3;i++){
			for (int j=0; j<3; j++){
				tmpCor.at<float>(bgdLabels.at<int>(row,0),i*3+j)+=
					(bgdData.at<float>(row,i)-bgdModel.at<float>(bgdLabels.at<int>(row,0),i))*
					(bgdData.at<float>(row,j)-bgdModel.at<float>(bgdLabels.at<int>(row,0),j));
			}
		}
		tmpCor.at<float>(bgdLabels.at<int>(row,0),9)+=1;
	}
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<9; j++){
			bgdModel.at<float>(i,3+j)=tmpCor.at<float>(i,j)/tmpCor.at<float>(i,9);
		}
	}
	//std::cout<<"fgdModel :"<<std::endl<<fgdModel<<std::endl;
	//std::cout<<"bgdModel :"<<std::endl<<bgdModel<<std::endl;
}

void GrabCut2D::GMMGamma(cv::InputArray _bgdModel, cv::InputArray _fgdModel, cv::InputArray _bgdData, cv::InputArray _fgdData,
	cv::InputOutputArray _bgdGamma, cv::InputOutputArray _fgdGamma, cv::InputOutputArray _bgdK, cv::InputOutputArray _fgdK)
{
	cv::Mat bgdModel = _bgdModel.getMat();
    cv::Mat fgdModel = _fgdModel.getMat();
	cv::Mat bgdData = _bgdData.getMat();
	cv::Mat fgdData = _fgdData.getMat();
	cv::Mat& bgdGamma = _bgdGamma.getMatRef();
	cv::Mat& fgdGamma = _fgdGamma.getMatRef();
	cv::Mat& bgdK = _bgdK.getMatRef();
	cv::Mat& fgdK = _fgdK.getMatRef();

	bgdGamma.create(bgdData.rows, NUM_CLUSTER, CV_32F);
	bgdK.create(bgdData.rows, 1, CV_32S);
	fgdGamma.create(fgdData.rows, NUM_CLUSTER, CV_32F);
	fgdK.create(fgdData.rows, 1, CV_32S);

	for( int num=0;num<fgdData.rows;num++){
		cv::Mat X=fgdData.row(num);
		for( int cluster=0;cluster<NUM_CLUSTER;cluster++){
			cv::Mat params=fgdModel.row(cluster);
			fgdGamma.at<float>(num,cluster)=Gauss(X,params);
		}
		cv::Mat Ps=fgdGamma.row(num);
		float pSum=0;
		for (int i=0;i<NUM_CLUSTER;i++)
			pSum+=Ps.at<float>(0,i);
		//std::cout<<"fgdGamma.row(num) :"<<fgdGamma.row(num)<<std::endl;
		Ps=Ps/pSum;
		//std::cout<<"Ps :"<<Ps<<std::endl;
		//std::cout<<"fgdGamma.row(num) :"<<fgdGamma.row(num)<<std::endl;
		float maxP=Ps.at<float>(0,0);
		int maxIndex=0;
		for (int i=1;i<NUM_CLUSTER;i++){
			if(Ps.at<float>(0,i)>maxP){
				maxP=Ps.at<float>(0,i);
				maxIndex=i;
			}
		}
		fgdK.at<int>(num,0)=maxIndex;
		//std::cout<<"fgdK.at<int>(num,0) :"<<fgdK.at<int>(num,0)<<std::endl;
	}

	for( int num=0;num<bgdData.rows;num++){
		cv::Mat X=bgdData.row(num);
		for( int cluster=0;cluster<NUM_CLUSTER;cluster++){
			cv::Mat params=bgdModel.row(cluster);
			bgdGamma.at<float>(num,cluster)=Gauss(X,params);
		}
		cv::Mat Ps=bgdGamma.row(num);
		float pSum=0;
		for (int i=0;i<NUM_CLUSTER;i++)
			pSum+=Ps.at<float>(0,i);
		Ps=Ps/pSum;
		//std::cout<<"Ps :"<<Ps<<std::endl;
		//std::cout<<"bgdGamma.row(num) :"<<bgdGamma.row(num)<<std::endl;
		float maxP=Ps.at<float>(0,0);
		int maxIndex=0;
		for (int i=1;i<NUM_CLUSTER;i++){
			if(Ps.at<float>(0,i)>maxP){
				maxP=Ps.at<float>(0,i);
				maxIndex=i;
			}
		}
		bgdK.at<int>(num,0)=maxIndex;
		//std::cout<<"bgdK.at<int>(num,0) :"<<bgdK.at<int>(num,0)<<std::endl;
	}
}

float GrabCut2D::Gauss(cv::InputArray _X, cv::InputArray _params)
{
	cv::Mat X = _X.getMat();
	cv::Mat params = _params.getMat();

	cv::Mat mean = params(cv::Range::all(),cv::Range(0,3));
	cv::Mat corr;
	corr.create(3, 3, CV_32F);
	corr.at<float>(0,0)=params.at<float>(3);corr.at<float>(0,1)=params.at<float>(4);corr.at<float>(0,2)=params.at<float>(5);
	corr.at<float>(1,0)=params.at<float>(6);corr.at<float>(1,1)=params.at<float>(7);corr.at<float>(1,2)=params.at<float>(8);
	corr.at<float>(2,0)=params.at<float>(9);corr.at<float>(2,1)=params.at<float>(10);corr.at<float>(2,2)=params.at<float>(11);
	cv::Mat corrInv = corr.inv();
	float pi = params.at<float>(0,12);

	//std::cout<<"X :"<<X<<std::endl;
	//std::cout<<"params :"<<params<<std::endl;
	//std::cout<<"mean :"<<mean<<std::endl;
	//std::cout<<"corr :"<<corr<<std::endl;
	//std::cout<<"corrInv :"<<corrInv<<std::endl;
	//std::cout<<corr*corrInv<<std::endl;
	
	//std::cout<<"pi :"<<pi<<std::endl;

	cv::Mat delta=X-mean;
	//std::cout<<"delta :"<<delta<<std::endl;
	float distance=0;
	float tmpDistance[3]={0,0,0};
	for (int col=0;col<3;col++){
		for (int row=0;row<3;row++){
			tmpDistance[col]+=delta.at<float>(0,row)*corrInv.at<float>(row,col);
		}
		//std::cout<<tmpDistance[col]<<std::endl;
	}

	for (int i=0;i<3;i++){
		distance+=tmpDistance[i]*delta.at<float>(0,i);
	}
	//std::cout<<"distance :"<<distance<<std::endl;

	//std::cout<<"det :"<<cv::determinant(corr)<<std::endl;
	//std::cout<<"std::exp(-1./2.*distance) :"<<std::exp(-1./2.*distance)<<std::endl;

	float p=(float)std::exp(-1./2.*distance)/
		(float)std::sqrt(std::fabs(cv::determinant(corr)))/
		(float)(2.*PI)/(float)std::sqrt(2.*PI);

	//std::cout<<"p :"<<p<<std::endl;

	return p*pi;
}

void GrabCut2D::GMMUpdate(cv::InputArray _img, cv::InputArray _mask,
	cv::InputArray _bgdGamma, cv::InputArray _fgdGamma,
	cv::InputArray _bgdK, cv::InputArray _fgdK,
	cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
	cv::InputArray _bgdData,cv::InputArray _fgdData)
{
	cv::Mat img = _img.getMat();
    cv::Mat mask = _mask.getMat();
	cv::Mat bgdGamma = _bgdGamma.getMat();
	cv::Mat fgdGamma = _fgdGamma.getMat();
	cv::Mat bgdK = _bgdK.getMat();
	cv::Mat fgdK = _fgdK.getMat();
    cv::Mat& bgdModel = _bgdModel.getMatRef();
    cv::Mat& fgdModel = _fgdModel.getMatRef();
	cv::Mat bgdData = _bgdData.getMat();
	cv::Mat fgdData = _fgdData.getMat();

	//std::cout<<mask.type()<<std::endl;

	fgdModel.create(NUM_CLUSTER, 13, CV_32F);
	bgdModel.create(NUM_CLUSTER, 13, CV_32F);

	cv::Mat tmpMeans=cv::Mat::zeros(NUM_CLUSTER, 4, CV_32F);

	for( int row=0; row<fgdData.rows; row++){
		for (int cluster=0;cluster<NUM_CLUSTER;cluster++){
			cv::Mat data=fgdData.row(row);
			for (int i=0;i<3;i++){
				tmpMeans.at<float>(cluster, i)+=data.at<float>(0,i)*fgdGamma.at<float>(row,cluster);
			}
		}
		tmpMeans.at<float>(fgdK.at<int>(row,0),3)+=1;
	}
	
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<3; j++){
			fgdModel.at<float>(i,j)=tmpMeans.at<float>(i,j)/tmpMeans.at<float>(i,3);
		}
		fgdModel.at<float>(i,12)=tmpMeans.at<float>(i,3)/fgdData.rows;
	}
	//std::cout<<"tmpMeans :"<<tmpMeans<<std::endl;
	//std::cout<<"fgdModel :"<<fgdModel<<std::endl;

	tmpMeans=cv::Mat::zeros(NUM_CLUSTER, 4, CV_32F);

	for( int row=0; row<bgdData.rows; row++){
		//std::cout<<"data :"<<bgdData.row(row)<<std::endl;
		//std::cout<<"gamma :"<<bgdGamma.row(row)<<std::endl;
		//std::cout<<"k :"<<bgdK.row(row)<<std::endl;
		for (int cluster=0;cluster<NUM_CLUSTER;cluster++){
			cv::Mat data=bgdData.row(row);
			for (int i=0;i<3;i++){
				tmpMeans.at<float>(cluster, i)+=data.at<float>(0,i)*bgdGamma.at<float>(row,cluster);
			}
		}
		//std::cout<<"tmpMeans :"<<tmpMeans<<std::endl;
		tmpMeans.at<float>(bgdK.at<int>(row,0),3)+=1;
	}
	
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<3; j++){
			bgdModel.at<float>(i,j)=tmpMeans.at<float>(i,j)/tmpMeans.at<float>(i,3);
		}
		bgdModel.at<float>(i,12)=tmpMeans.at<float>(i,3)/bgdData.rows;
	}
	//std::cout<<bgdModel<<std::endl;

	
	cv::Mat tmpCor=cv::Mat::zeros(NUM_CLUSTER, 10, CV_32F);

	for( int row=0; row<fgdData.rows; row++){
		//std::cout<<"data :"<<fgdData.row(row)<<std::endl;
		//std::cout<<"fgdModel :"<<fgdModel<<std::endl;
		//std::cout<<"gamma :"<<fgdGamma.row(row)<<std::endl;
		//std::cout<<"k :"<<fgdK.row(row)<<std::endl;
		for (int cluster=0;cluster<NUM_CLUSTER;cluster++){
			for (int i=0;i<3;i++){
				for (int j=0; j<3; j++){
					tmpCor.at<float>(cluster,i*3+j)+=
						fgdGamma.at<float>(row,cluster)*
						(fgdData.at<float>(row,i)-fgdModel.at<float>(fgdK.at<int>(row,0),i))*
						(fgdData.at<float>(row,j)-fgdModel.at<float>(fgdK.at<int>(row,0),j));
				}
			}
		}
		//std::cout<<"tmpCor :"<<tmpCor<<std::endl;
		tmpCor.at<float>(fgdK.at<int>(row,0),9)+=1;
	}
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<9; j++){
			fgdModel.at<float>(i,3+j)=tmpCor.at<float>(i,j)/tmpCor.at<float>(i,9);
		}
	}
	
	//std::cout<<"tmpCor :"<<std::endl<<tmpCor<<std::endl;
	//std::cout<<"fgdModel :"<<fgdModel<<std::endl;

	tmpCor=cv::Mat::zeros(NUM_CLUSTER, 10, CV_32F);

	for( int row=0; row<bgdData.rows; row++){
		for (int cluster=0;cluster<NUM_CLUSTER;cluster++){
			for (int i=0;i<3;i++){
				for (int j=0; j<3; j++){
					tmpCor.at<float>(cluster,i*3+j)+=
						bgdGamma.at<float>(row,cluster)*
						(bgdData.at<float>(row,i)-bgdModel.at<float>(bgdK.at<int>(row,0),i))*
						(bgdData.at<float>(row,j)-bgdModel.at<float>(bgdK.at<int>(row,0),j));
				}
			}
		}
		tmpCor.at<float>(bgdK.at<int>(row,0),9)+=1;
	}
	for( int i=0; i<NUM_CLUSTER; i++){
		for( int j=0; j<9; j++){
			bgdModel.at<float>(i,3+j)=tmpCor.at<float>(i,j)/tmpCor.at<float>(i,9);
		}
	}
	
	//std::cout<<"tmpCor :"<<std::endl<<tmpCor<<std::endl;
	//std::cout<<"bgdModel :"<<bgdModel<<std::endl;
}

void GrabCut2D::GMMUpdateData(cv::InputArray _img, cv::InputArray _mask,
	cv::InputOutputArray _bgdData,cv::InputOutputArray _fgdData)
{
	cv::Mat img = _img.getMat();
    cv::Mat mask = _mask.getMat();
	cv::Mat& bgdData = _bgdData.getMatRef();
	cv::Mat& fgdData = _fgdData.getMatRef();

	int numFgd=0;
	int numBgd=0;

	for( int row=0; row<mask.rows; row++){
		for( int col=0; col<mask.cols; col++){
			if	(mask.at<uchar>(row,col)%2==1){
				numFgd++;
			}else{
				numBgd++;
			}
		}
	}

	fgdData.create(numFgd,3,CV_32F);
	bgdData.create(numBgd,3,CV_32F);

	numFgd=0;
	numBgd=0;
	for( int row=0; row<mask.rows; row++){
		for( int col=0; col<mask.cols; col++){
			if	(mask.at<uchar>(row,col)%2==1){
				for( int z=0; z<3; z++){
					fgdData.at<float>(numFgd, z) = img.at<cv::Vec3b>(row,col)[z];
				}
				numFgd++;
			}else{
				for( int z=0; z<3; z++){
					bgdData.at<float>(numBgd, z) = img.at<cv::Vec3b>(row,col)[z];
				}
				numBgd++;
			}
		}
	}
}