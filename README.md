# Finding-crop-image
Finding crop image

I used 2 different programming language. 
  - Main language ı use is Python. I am not good at c++ language as well as in python. (You gonna understand when look at your c++ code but i can learn easily)
  -I create 4 different algorithms. 3 of them about just crop image, other one about crop and crop_rotate image. 
  -Also i added comment to many lines for description.
  

Explain the 4 function in code:

1) templateMatchingOpencvPreparedFunctions
    - Here the main idea is that write the program as fast as possible.
    - I used matchTmeplate function then take the location of matching object.
    
2) templateMatchingEqual
    -Here the main idea is that slide the object pixel by pixel in main image then compare two image are equal or not in grayscale.
    -...
    
3) templateMatchingMean
    - Here the main idea is that slide the object pixel by pixel in main image and take the differences between two image and take the mean of matrix.
    - Smallest mean value can our object.
    
4) featureMatchingSirf
    -Here the main idea is that use feature extraction ob object and pair this features in main image. If any matching greater than 5 (we determine this value) we found object.
    -Then using findHomograpy we calculate the coordinates of object.
    -I write this code in python easily but i couldnt in c++. 
    
    
