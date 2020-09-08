
# Seam Carving Using openCV and CUDA

## Concept
Seam carving is an algorithm for content-aware image resizing. The algorithm reduces its width/height by removing seams(path of least importance) in an image. This prevents undesirable distortions to objects of importance. Performing the algorithm sequentially takes a long time as each pixel has to be calculated one at a time. Hence, using CUDA to parallelize the algorithm results in a faster speed.<br />
The aim of this project is to determine how much faster does parallelizing the algorithm take compared to sequentially computing the algorithm.<br />
For more information: https://en.wikipedia.org/wiki/Seam_carving<br />

## Build and run
Visual Studio with CUDA and openCV (with CUDA compatability) required.<br />
To run the project, open the solution in Visual Studio.<br />
To select image and width/height reduction amount, edit within the main functions of both the sequential and parallel projects.<br />
Width reduction = reduceWidth<br />
Height reduction = reduceHeight<br />
The solution can be run in Debug mode. To run in Release mode, openCV must be included and linked in the properties.<br />

## Project structure

```
.
seam_carving
│   
│   seam_carving.sln
|
└───images
│   │   List of input images
│   
└───parallel
│   │   kernel.cu
|   |   parallel.cpp
|   |   parallel.h
|   |   parallel.vcxproj
│   
└───sequential
|   |   sequential.cpp
|   |   sequential.vcxproj
|   |   sequential.vcxproj.filters
```


## Parallel with CUDA
  * **Main** <br />
    Main function where the user can change the reduction amount and the input image. Outputs the image after seam carving and the time taken for each main function call.<br />
    
  * **Create energy image** <br />
    Given an input image, the function createEnergyImg(Mat& image) will convert it to grayscale and apply a sobel filter to the grayscale image. openCV cuda library is used to parallelize the function. Returns the energy of the image.

  * **Create energy map** <br />
    Creates a cumulative energy map. The function getEnergyMap(energy, energyMap, rowSize, colSize, seamDirection) is called, which will initialize and call the CUDA function to get the energy map. In the kernel function, threads will be created for each row or column, depending on the seam direction. The minimum cumulative energy will be created row by row or column by column depending on if the seam is vertical or horizontal respectively. Each thread will compute the minimum cumulative energy for one pixel in the row/column by taking the minimum cumulative energy of it's above neighbor.<br />
    
  * **Find optimal seam** <br />
    The optimal seam is found by starting from the minimum cumulative energy of the last row/column. CUDA is used via parallel reduction to find the location of the last row/column's minimum cumulative energy.  Starting from the last row's minimum cumulative energy, the function backtracks to find the minimum cumulative energy of the previous row/column neighbor. This results in a seam being formed.<br />

  * **Remove seam** <br />
    After finding the optimal seam, the function removeSeam creates a new output image that leaves out the pixels corresponding to the seam, resulting in a new image that is reduced by one row/column depending on the seam direction.<br />  



## Sequential

* Same concept as parallel but using sequential code to compute the energy, cumulative energy map, optimal seam, and to remove the seam from the image.   <br />



## Results
### Hardware
| | Component | Component used |
|--|--|--|
|1.| Processor |  Intel Core i5-4690k CPU @ 3.50GHz |
|2.|Memory |  8GB |
|3.| GPU | NVIDIA GTX970 |

### inputPrague.jpg

### inputBroadwayTower.jpg

### inputColdplayWings.jpg


### inputGoldenGate.jpg



## Future changes
* Content aware resizing to prevent undesirable removal of objects, such as faces.<br />
* Ability to increase width or height of image.<br />



## Reference
* http://abhandaru.github.io/gpu-seamcarving/final-report.pdf<br />
* https://github.com/kalpeshdusane/Seam-Carving-B.E.-Project<br />
* https://github.com/davidshower/seam-carving<br />
