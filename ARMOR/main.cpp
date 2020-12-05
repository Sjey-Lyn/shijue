#include "thread"
#include <iostream>
#include "ImageConsProd.h"

int main()
{

    // start threads
    ImageConsProd image_cons_prod;
//    image_cons_prod.ImageProducer();
//     利用多线程加快速度
    std::thread t1(&ImageConsProd::ImageProducer, image_cons_prod); // pass by reference
    std::thread t2(&ImageConsProd::ImageConsumer, std::ref(image_cons_prod));

    t1.join();
    t2.join();

}
