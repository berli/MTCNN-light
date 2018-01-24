#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <sys/time.h>

long getMillSeconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return 1000*tv.tv_sec+tv.tv_usec/1000;
}


int main(int argc, char*argv[])
{
    Mat image;
    VideoCapture cap;
    if(argc == 1)
    {
       bool lbRet =  cap.open("../../hls-20.mp4");
       if(!lbRet)
       {
          lbRet = cap.open(0);
          cout<<"fail to open!"<<endl;
       }
    }
    else if(argc ==2 )
    {
       image = imread(argv[1]);
       mtcnn find(image.rows, image.cols);
       long s = getMillSeconds();

       vector<FaceInfo>vecFace;
       int liRet = find.findFace(image, vecFace);

       namedWindow("result", CV_WINDOW_AUTOSIZE);
       imshow("result", image);
       imwrite("result.jpg",image);
       cout<<"faces:"<<vecFace.size()<<" ret:"<<liRet<<" time is  "<<(getMillSeconds()-s)<<" ms"<<endl;
       waitKey(0);
       image.release();
       return 0;
  }

  cap>>image;
  if(!image.data)
  {
      cout<<"读取视频失败"<<endl;
      return -1;
  }

  mtcnn find(image.rows, image.cols);
  vector<FaceInfo>vecFace;
  while(true)
  {
  long s = getMillSeconds();
      cap>>image;
      find.findFace(image, vecFace);
	int liFps = getMillSeconds() - s;
    if( liFps > 0 )
        liFps = 1000/liFps;
    else
        liFps = 1000;
    
    string lsFps = "fps:";
    lsFps += to_string(liFps);
    lsFps += " time is:";
    lsFps += to_string(getMillSeconds()-s);
    cv::putText(image, lsFps, cvPoint(3, 13),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 255, 0), 1, CV_AA);
      imshow("result", image);
      if( waitKey(1)>=0 ) 
		  break;
  }

  waitKey(0);
  image.release();
  return 0;
}
