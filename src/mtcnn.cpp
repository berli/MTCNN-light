#include "mtcnn.h"

Pnet::Pnet()
{
    Pthreshold = 0.6;
    nms_threshold = 0.5;
    firstFlag = true;
    rgb = new pBox;

    conv1_matrix = new pBox;
    conv1 = new pBox;
    maxPooling1 = new pBox;

    maxPooling_matrix = new pBox;
    conv2 = new pBox;

    conv3_matrix = new pBox;
    conv3 = new pBox;

    score_matrix = new pBox;
    score_ = new pBox;

    location_matrix = new pBox;
    location_ = new pBox;

    conv1_wb = new Weight;
    prelu_gmma1 = new pRelu;
    conv2_wb = new Weight;
    prelu_gmma2 = new pRelu;
    conv3_wb = new Weight;
    prelu_gmma3 = new pRelu;
    conv4c1_wb = new Weight;
    conv4c2_wb = new Weight;
    //                                 w       sc  lc ks s  p
    long conv1 = initConvAndFc(conv1_wb, 10, 3, 3, 1, 0);
    initpRelu(prelu_gmma1, 10);
    long conv2 = initConvAndFc(conv2_wb, 16, 10, 3, 1, 0);
    initpRelu(prelu_gmma2, 16);
    long conv3 = initConvAndFc(conv3_wb, 32, 16, 3, 1, 0);
    initpRelu(prelu_gmma3, 32);
    long conv4c1 = initConvAndFc(conv4c1_wb, 2, 32, 1, 1, 0);
    long conv4c2 = initConvAndFc(conv4c2_wb, 4, 32, 1, 1, 0);
    long dataNumber[13] = {conv1,10,10, conv2,16,16, conv3,32,32, conv4c1,2, conv4c2,4};
    mydataFmt *pointTeam[13] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4c1_wb->pdata, conv4c1_wb->pbias, \
                            conv4c2_wb->pdata, conv4c2_wb->pbias \
                            };
    string filename = "Pnet.txt";
    readData(filename, dataNumber, pointTeam);
}

Pnet::~Pnet()
{
    freepBox(rgb);
    freepBox(conv1);
    freepBox(maxPooling1);
    freepBox(conv2);
    freepBox(conv3);
    freepBox(score_);
    freepBox(location_);

    freepBox(conv1_matrix);
    freeWeight(conv1_wb);
    freepRelu(prelu_gmma1);
    freepBox(maxPooling_matrix);
    freeWeight(conv2_wb);
    freepBox(conv3_matrix);
    freepRelu(prelu_gmma2);
    freeWeight(conv3_wb);
    freepBox(score_matrix);
    freepRelu(prelu_gmma3);
    freeWeight(conv4c1_wb);
    freepBox(location_matrix);
    freeWeight(conv4c2_wb);
}

void Pnet::run(Mat &image, float scale)
{
    if(firstFlag)
    {
        image2MatrixInit(image, rgb);

        feature2MatrixInit(rgb, conv1_matrix, conv1_wb);
        convolutionInit(conv1_wb, rgb, conv1, conv1_matrix);

        maxPoolingInit(conv1, maxPooling1, 2, 2);
        feature2MatrixInit(maxPooling1, maxPooling_matrix, conv2_wb);
        convolutionInit(conv2_wb, maxPooling1, conv2, maxPooling_matrix);

        feature2MatrixInit(conv2, conv3_matrix, conv3_wb);
        convolutionInit(conv3_wb, conv2, conv3, conv3_matrix);

        feature2MatrixInit(conv3, score_matrix, conv4c1_wb);
        convolutionInit(conv4c1_wb, conv3, score_, score_matrix);

        feature2MatrixInit(conv3, location_matrix, conv4c2_wb);
        convolutionInit(conv4c2_wb, conv3, location_, location_matrix);
        firstFlag = false;
    }

    image2Matrix(image, rgb);

    feature2Matrix(rgb, conv1_matrix, conv1_wb);
    convolution(conv1_wb, rgb, conv1, conv1_matrix);
    prelu(conv1, conv1_wb->pbias, prelu_gmma1->pdata);
    //Pooling layer
    maxPooling(conv1, maxPooling1, 2, 2);

    feature2Matrix(maxPooling1, maxPooling_matrix, conv2_wb);
    convolution(conv2_wb, maxPooling1, conv2, maxPooling_matrix);
    prelu(conv2, conv2_wb->pbias, prelu_gmma2->pdata);
    //conv3
    feature2Matrix(conv2, conv3_matrix, conv3_wb);
    convolution(conv3_wb, conv2, conv3, conv3_matrix);
    prelu(conv3, conv3_wb->pbias, prelu_gmma3->pdata);
    //conv4c1   score
    feature2Matrix(conv3, score_matrix, conv4c1_wb);
    convolution(conv4c1_wb, conv3, score_, score_matrix);
    addbias(score_, conv4c1_wb->pbias);
    softmax(score_);
    // pBoxShow(score_);

    //conv4c2   location
    feature2Matrix(conv3, location_matrix, conv4c2_wb);
    convolution(conv4c2_wb, conv3, location_, location_matrix);
    addbias(location_, conv4c2_wb->pbias);
    //softmax layer
    generateBbox(score_, location_, scale);
}

void Pnet::generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt& scale)
{
    //for pooling
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    mydataFmt *p = score->pdata + score->width*score->height;
    mydataFmt *plocal = location->pdata;
    struct Bbox bbox;
    struct orderScore order;
    for(int row=0;row<score->height;row++)
    {
        for(int col=0;col<score->width;col++)
	{
            if(*p>Pthreshold)
	    {
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*row+1)/scale);
                bbox.y1 = round((stride*col+1)/scale);
                bbox.x2 = round((stride*row+1+cellsize)/scale);
                bbox.y2 = round((stride*col+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=*(plocal+channel*location->width*location->height);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}

Rnet::Rnet()
{
    Rthreshold = 0.7;

    rgb = new pBox;
    conv1_matrix = new pBox;
    conv1_out = new pBox;
    pooling1_out = new pBox;

    conv2_matrix = new pBox;
    conv2_out = new pBox;
    pooling2_out = new pBox;

    conv3_matrix = new pBox;
    conv3_out = new pBox;

    fc4_out = new pBox;

    score_ = new pBox;
    location_ = new pBox;

    conv1_wb = new Weight;
    prelu_gmma1 = new pRelu;
    conv2_wb = new Weight;
    prelu_gmma2 = new pRelu;
    conv3_wb = new Weight;
    prelu_gmma3 = new pRelu;
    fc4_wb = new Weight;
    prelu_gmma4 = new pRelu;
    score_wb = new Weight;
    location_wb = new Weight;
    // //                             w         sc  lc ks s  p
    long conv1 = initConvAndFc(conv1_wb, 28, 3, 3, 1, 0);
    initpRelu(prelu_gmma1, 28);
    long conv2 = initConvAndFc(conv2_wb, 48, 28, 3, 1, 0);
    initpRelu(prelu_gmma2, 48);
    long conv3 = initConvAndFc(conv3_wb, 64, 48, 2, 1, 0);
    initpRelu(prelu_gmma3, 64);
    long fc4 = initConvAndFc(fc4_wb, 128, 576, 1, 1, 0);
    initpRelu(prelu_gmma4, 128);
    long score = initConvAndFc(score_wb, 2, 128, 1, 1, 0);
    long location = initConvAndFc(location_wb, 4, 128, 1, 1, 0);
    long dataNumber[16] = {conv1,28,28, conv2,48,48, conv3,64,64, fc4,128,128, score,2, location,4};
    mydataFmt *pointTeam[16] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                                conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                                conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                                fc4_wb->pdata, fc4_wb->pbias, prelu_gmma4->pdata, \
                                score_wb->pdata, score_wb->pbias, \
                                location_wb->pdata, location_wb->pbias \
                                };
    string filename = "Rnet.txt";
    readData(filename, dataNumber, pointTeam);

    //Init the network
    RnetImage2MatrixInit(rgb);
    feature2MatrixInit(rgb, conv1_matrix, conv1_wb);
    convolutionInit(conv1_wb, rgb, conv1_out, conv1_matrix);
    maxPoolingInit(conv1_out, pooling1_out, 3, 2);
    feature2MatrixInit(pooling1_out, conv2_matrix, conv2_wb);
    convolutionInit(conv2_wb, pooling1_out, conv2_out, conv2_matrix);
    maxPoolingInit(conv2_out, pooling2_out, 3, 2);
    feature2MatrixInit(pooling2_out, conv3_matrix, conv3_wb);
    convolutionInit(conv3_wb, pooling2_out, conv3_out, conv3_matrix);
    fullconnectInit(fc4_wb, fc4_out);
    fullconnectInit(score_wb, score_);
    fullconnectInit(location_wb, location_);
}

Rnet::~Rnet()
{
    freepBox(rgb);
    freepBox(conv1_matrix);
    freepBox(conv1_out);
    freepBox(pooling1_out);
    freepBox(conv2_matrix);
    freepBox(conv2_out);
    freepBox(pooling2_out);
    freepBox(conv3_matrix);
    freepBox(conv3_out);
    freepBox(fc4_out);
    freepBox(score_);
    freepBox(location_);

    freeWeight(conv1_wb);
    freepRelu(prelu_gmma1);
    freeWeight(conv2_wb);
    freepRelu(prelu_gmma2);
    freeWeight(conv3_wb);
    freepRelu(prelu_gmma3);
    freeWeight(fc4_wb);
    freepRelu(prelu_gmma4);
    freeWeight(score_wb);
    freeWeight(location_wb);
}

void Rnet::RnetImage2MatrixInit(struct pBox *pbox)
{
    pbox->channel = 3;
    pbox->height = 24;
    pbox->width = 24;

    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}

void Rnet::run(Mat &image)
{
    image2Matrix(image, rgb);

    feature2Matrix(rgb, conv1_matrix, conv1_wb);
    convolution(conv1_wb, rgb, conv1_out, conv1_matrix);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    maxPooling(conv1_out, pooling1_out, 3, 2);

    feature2Matrix(pooling1_out, conv2_matrix, conv2_wb);
    convolution(conv2_wb, pooling1_out, conv2_out, conv2_matrix);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);
    maxPooling(conv2_out, pooling2_out, 3, 2);

    //conv3
    feature2Matrix(pooling2_out, conv3_matrix, conv3_wb);
    convolution(conv3_wb, pooling2_out, conv3_out, conv3_matrix);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    //flatten
    fullconnect(fc4_wb, conv3_out, fc4_out);
    prelu(fc4_out, fc4_wb->pbias, prelu_gmma4->pdata);

    //conv51   score
    fullconnect(score_wb, fc4_out, score_);
    addbias(score_, score_wb->pbias);
    softmax(score_);

    //conv5_2   location
    fullconnect(location_wb, fc4_out, location_);
    addbias(location_, location_wb->pbias);
    // pBoxShow(location_);
}

Onet::Onet()
{
    Othreshold = 0.8;
    rgb = new pBox;

    conv1_matrix = new pBox;
    conv1_out = new pBox;
    pooling1_out = new pBox;

    conv2_matrix = new pBox;
    conv2_out = new pBox;
    pooling2_out = new pBox;

    conv3_matrix = new pBox;
    conv3_out = new pBox;
    pooling3_out = new pBox;

    conv4_matrix = new pBox;
    conv4_out = new pBox;

    fc5_out = new pBox;

    score_ = new pBox;
    location_ = new pBox;
    keyPoint_ = new pBox;

    conv1_wb = new Weight;
    prelu_gmma1 = new pRelu;
    conv2_wb = new Weight;
    prelu_gmma2 = new pRelu;
    conv3_wb = new Weight;
    prelu_gmma3 = new pRelu;
    conv4_wb = new Weight;
    prelu_gmma4 = new pRelu;
    fc5_wb = new Weight;
    prelu_gmma5 = new pRelu;
    score_wb = new Weight;
    location_wb = new Weight;
    keyPoint_wb = new Weight;

    // //                             w        sc  lc ks s  p
    long conv1 = initConvAndFc(conv1_wb, 32, 3, 3, 1, 0);
    initpRelu(prelu_gmma1, 32);
    long conv2 = initConvAndFc(conv2_wb, 64, 32, 3, 1, 0);
    initpRelu(prelu_gmma2, 64);
    long conv3 = initConvAndFc(conv3_wb, 64, 64, 3, 1, 0);
    initpRelu(prelu_gmma3, 64);
    long conv4 = initConvAndFc(conv4_wb, 128, 64, 2, 1, 0);
    initpRelu(prelu_gmma4, 128);
    long fc5 = initConvAndFc(fc5_wb, 256, 1152, 1, 1, 0);
    initpRelu(prelu_gmma5, 256);
    long score = initConvAndFc(score_wb, 2, 256, 1, 1, 0);
    long location = initConvAndFc(location_wb, 4, 256, 1, 1, 0);
    long keyPoint = initConvAndFc(keyPoint_wb, 10, 256, 1, 1, 0);
    long dataNumber[21] = {conv1,32,32, conv2,64,64, conv3,64,64, conv4,128,128, fc5,256,256, score,2, location,4, keyPoint,10};
    mydataFmt *pointTeam[21] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                                conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                                conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                                conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                                fc5_wb->pdata, fc5_wb->pbias, prelu_gmma5->pdata, \
                                score_wb->pdata, score_wb->pbias, \
                                location_wb->pdata, location_wb->pbias, \
                                keyPoint_wb->pdata, keyPoint_wb->pbias \
                                };
    string filename = "Onet.txt";
    readData(filename, dataNumber, pointTeam);

    //Init the network
    OnetImage2MatrixInit(rgb);

    feature2MatrixInit(rgb, conv1_matrix, conv1_wb);
    convolutionInit(conv1_wb, rgb, conv1_out, conv1_matrix);
    maxPoolingInit(conv1_out, pooling1_out, 3, 2);

    feature2MatrixInit(pooling1_out, conv2_matrix, conv2_wb);
    convolutionInit(conv2_wb, pooling1_out, conv2_out, conv2_matrix);
    maxPoolingInit(conv2_out, pooling2_out, 3, 2);

    feature2MatrixInit(pooling2_out, conv3_matrix, conv3_wb);
    convolutionInit(conv3_wb, pooling2_out, conv3_out, conv3_matrix);
    maxPoolingInit(conv3_out, pooling3_out, 2, 2);

    feature2MatrixInit(pooling3_out, conv4_matrix, conv4_wb);
    convolutionInit(conv4_wb, pooling3_out, conv4_out, conv4_matrix);

    fullconnectInit(fc5_wb, fc5_out);
    fullconnectInit(score_wb, score_);
    fullconnectInit(location_wb, location_);
    fullconnectInit(keyPoint_wb, keyPoint_);
}

Onet::~Onet()
{
    freepBox(rgb);
    freepBox(conv1_matrix);
    freepBox(conv1_out);
    freepBox(pooling1_out);
    freepBox(conv2_matrix);
    freepBox(conv2_out);
    freepBox(pooling2_out);
    freepBox(conv3_matrix);
    freepBox(conv3_out);
    freepBox(pooling3_out);
    freepBox(conv4_matrix);
    freepBox(conv4_out);
    freepBox(fc5_out);
    freepBox(score_);
    freepBox(location_);
    freepBox(keyPoint_);

    freeWeight(conv1_wb);
    freepRelu(prelu_gmma1);
    freeWeight(conv2_wb);
    freepRelu(prelu_gmma2);
    freeWeight(conv3_wb);
    freepRelu(prelu_gmma3);
    freeWeight(conv4_wb);
    freepRelu(prelu_gmma4);
    freeWeight(fc5_wb);
    freepRelu(prelu_gmma5);
    freeWeight(score_wb);
    freeWeight(location_wb);
    freeWeight(keyPoint_wb);
}

void Onet::OnetImage2MatrixInit(struct pBox *pbox)
{
    pbox->channel = 3;
    pbox->height = 48;
    pbox->width = 48;

    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Onet::run(Mat &image)
{
    image2Matrix(image, rgb);

    feature2Matrix(rgb, conv1_matrix, conv1_wb);
    convolution(conv1_wb, rgb, conv1_out, conv1_matrix);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    //Pooling layer
    maxPooling(conv1_out, pooling1_out, 3, 2);

    feature2Matrix(pooling1_out, conv2_matrix, conv2_wb);
    convolution(conv2_wb, pooling1_out, conv2_out, conv2_matrix);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);
    maxPooling(conv2_out, pooling2_out, 3, 2);

    //conv3
    feature2Matrix(pooling2_out, conv3_matrix, conv3_wb);
    convolution(conv3_wb, pooling2_out, conv3_out, conv3_matrix);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);
    maxPooling(conv3_out, pooling3_out, 2, 2);

    //conv4
    feature2Matrix(pooling3_out, conv4_matrix, conv4_wb);
    convolution(conv4_wb, pooling3_out, conv4_out, conv4_matrix);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);

    fullconnect(fc5_wb, conv4_out, fc5_out);
    prelu(fc5_out, fc5_wb->pbias, prelu_gmma5->pdata);

    //conv6_1   score
    fullconnect(score_wb, fc5_out, score_);
    addbias(score_, score_wb->pbias);
    softmax(score_);
    // pBoxShow(score_);

    //conv6_2   location
    fullconnect(location_wb, fc5_out, location_);
    addbias(location_, location_wb->pbias);
    // pBoxShow(location_);

    //conv6_2   location
    fullconnect(keyPoint_wb, fc5_out, keyPoint_);
    addbias(keyPoint_, keyPoint_wb->pbias);
    // pBoxShow(keyPoint_);
}


mtcnn::mtcnn(const int& row, const int &col)
{
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;

    float minl = row>col?row:col;
    int MIN_DET_SIZE = 12;
    int minsize = 60;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;

    while(minl>MIN_DET_SIZE)
    {
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    float minside = row<col ? row : col;
    int count = 0;
    for (vector<float>::iterator it = scales_.begin(); it != scales_.end(); it++)
    {
        if (*it > 1)
	{
            cout << "the minsize is too small" << endl;
            while (1);
        }
        if (*it < (MIN_DET_SIZE / minside))
	{
            scales_.resize(count);
            break;
        }
        count++;
    }
    simpleFace_ = new Pnet[scales_.size()];
}

mtcnn::~mtcnn()
{
    delete []simpleFace_;
}


int mtcnn::alignFace(const Mat &image, const vector<FaceInfo>&vecInfo, vector<Mat>&vecFaces)
{
    for(auto&m:vecInfo)
    {
          Mat faceROI = image(m.faceRect).clone();
      imshow("faceROI", faceROI);
          if(m.vecPts.size() != 5)
            continue;//5 points is required
          Point2f lefteye(m.vecPts[0]);
          Point2f righteye(m.vecPts[1]);

          Point2f center = Point2f((lefteye.x + righteye.x)*0.5, (lefteye.y + righteye.y)*0.5);  //两眼的中心点

	  double dy = righteye.y - lefteye.y;
	  double dx = righteye.x - lefteye.x;
	  double angle = atan2(dy, dx)*180.0 / CV_PI;     //角度
	  cout<<"dx:"<<dx<<" dy:"<<dy<<" angle:"<<angle<<endl;

	  Mat rot_mat = getRotationMatrix2D(center, angle, 1.0);   //求得仿射矩阵
	  Mat rot;         //变换后的人脸图像
	  warpAffine(faceROI, rot, rot_mat, faceROI.size());    //仿射变换
	  vecFaces.push_back(rot);
    }
}

int mtcnn::findFace(const Mat &image, vector<FaceInfo>&vecFaceInfo)
{
    struct orderScore order;
    int count = 0;
    for (size_t i = 0; i < scales_.size(); i++)
    {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
        simpleFace_[i].run(reImage, scales_.at(i));
        nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);

        for(vector<struct Bbox>::iterator it=simpleFace_[i].boundingBox_.begin(); it!=simpleFace_[i].boundingBox_.end();it++)
	{
            if((*it).exist)
	    {
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        simpleFace_[i].bboxScore_.clear();
        simpleFace_[i].boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)
        return -1;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols);

    //second stage
    count = 0;
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++)
    {
        if((*it).exist)
	{
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat secImage;
            resize(image(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR);
            refineNet.run(secImage);
            if(*(refineNet.score_->pdata+1)>refineNet.Rthreshold)
	    {
                memcpy(it->regreCoord, refineNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(refineNet.score_->pdata+1);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else
	    {
                (*it).exist=false;
            }
        }
    }
    if(count<1)
       return -2;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols);

    //third stage
    count = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++)
    {
        if((*it).exist)
        {
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat thirdImage;
            resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
            outNet.run(thirdImage);
            mydataFmt *pp=NULL;

            if(*(outNet.score_->pdata+1)>outNet.Othreshold)
            {
                memcpy(it->regreCoord, outNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(outNet.score_->pdata+1);
                pp = outNet.keyPoint_->pdata;
                for(int num=0;num<5;num++)
                {
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
                }
                for(int num=0;num<5;num++)
                {
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else
            {
                it->exist=false;
            }
        }
    }

    if(count<1)
      return -3;

    refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");

    bool lbHave = false;

    vecFaceInfo.clear();
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++)
    {
        if((*it).exist)
	{
            FaceInfo lFace;
            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
            Rect lrect(Point((*it).y1,(*it).x1),Point((*it).y2,(*it).x2));
            lFace.faceRect = lrect;
            for(int num=0;num<5;num++)
            {
                Point lpt( (int)*(it->ppoint+num), (int)*(it->ppoint+num+5) );
                circle(image,lpt,3,Scalar(0,255,255), -1);
                lFace.vecPts.push_back(lpt);
            }
             vecFaceInfo.push_back(lFace);

	    lbHave = true;
        }
    }

    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();

    return (lbHave?0:-1);
}

