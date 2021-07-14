#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include "hnswlib/space_l2.h"



#define EPS 1e-6
bool close_enough(double a,double b)
{
  if (std::abs(a-b) < EPS)
    return true;

  if (std::abs(a-b)/std::max(std::abs(a),std::abs(b)) < EPS)
    return true;

  return false;
}



int main()
{
  unsigned char a[1024];
  unsigned char b[1024];


  float af[1024];
  float bf[1024];


  int (*ssdip)(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) = NULL;
  float (*ssdfp)(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) = NULL;

  float (*ipfp)(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) = NULL;



  for (int y = 0; y < 1000000; y++)
  {
    size_t len = (y % 1000);
    if (len == 0) {
      for (int i=0;i<1024;i++)
      {
        af[i] = (a[i] = rand() % 255);
        bf[i] = (b[i] = rand() % 255);
      }
    }
    hnswlib::L2SpaceI l2(len);
    hnswlib::L2Space l2f(len);
    hnswlib::InnerProductSpace ips(len);

    ssdip = l2.get_dist_func();

    int ssdrefi = hnswlib::L2SqrI(a,b,&len);
    int ssdopti = ssdip(a,b,&len);

    if (ssdrefi != ssdopti)
    {
      printf("error %d %d %d\n",y,ssdrefi,ssdopti);
      return -1;
    }
    ssdfp = l2f.get_dist_func();

    double ssdreff = hnswlib::L2Sqr(af,bf,&len);
    double ssdoptf = ssdfp(af,bf,&len);

    if (!close_enough(ssdreff,ssdoptf))
    {
      printf("error float %d %lf %lf\n",y,ssdreff,ssdoptf);
      return -1;
    }

    ipfp = ips.get_dist_func();

    double ip_ref = hnswlib::InnerProduct(af,bf,&len);
    double ip_lib = ipfp(af,bf,&len);

    if (!close_enough(ip_ref , ip_lib))
    {
      printf("error inner product %d %lf %lf\n",y,ip_ref,ip_lib);
      return -1;
    }


  }

  printf("Done with no errors\n");
  return 0;
}

