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
    if(argc ==2 )
    {
    Mat image = imread(argv[1]);
    mtcnn find(image.rows, image.cols);
    long s = getMillSeconds();

    vector<FaceInfo>vecFace;
    int liRet = find.findFace(image, vecFace);

    namedWindow("result", CV_WINDOW_AUTOSIZE);
    imshow("result", image);
    imwrite("result.jpg",image);
    cout<<"ret:"<<liRet<<" time is  "<<(getMillSeconds()-s)/10e3<<" ms"<<endl;
    waitKey(0);
    image.release();
    return 0;
  }

  Mat image;
  VideoCapture cap(0);
  if(!cap.isOpened())
      cout<<"fail to open!"<<endl;
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
      imshow("result", image);
      if( waitKey(1)>=0 ) break;
      s = getMillSeconds() -s;
      cout<<"time is  "<<s<<endl;
  }

  waitKey(0);
  image.release();
  return 0;
}
