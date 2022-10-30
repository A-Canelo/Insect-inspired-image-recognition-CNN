# Insect-inspired-image-recognition-CNN
An image recognition Deep Learning model based on the visual system of fruit fly *Drosophila*, *FlyDrosNet*, for embedding on a *crazyflie 2.1* drone STM32 and *ai-deck* GAP8 to perform an autonomous flight. 

*FlyDrosNet* is a CNN based on the visual system of the fly *Drosophila*. The architecture uses the neural pathways necessary for feature and looming detection. It has 3 outputs to classify images in 3 categories (Collision, Rectangle, Square).

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/FlyDrosNet_diagram.png" width=60% height=60%>

We also provide a pattern dataset labeled as mentioned, and compared the performance results to other relevant architectures, showing that we have achived a sufficient accuracy using low memory, which is essential for embedded deployment.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/pattern_dataset_sample.png" width=50% height=50%>

| Architecture  | Top accuracy (%) | Parameters (#) | Memory (KB) |
| :---: | :---: | :---: | :---: |
| ResNet101  | 97.66  | 42,658,051 | 489,290 |
| MobileNetV2  | 96.66  | 2,261,251 | 26,450 |
| FlyDrosNet  | **95.33**  | 747,665 | **8,968** |
| FlyDrosNet_8bit  | **84.00**  | 747,665 | **753** |

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/performance_comparison.png" width=60% height=60%>

For embedding *FlyDrosNet* on the *ai-deck* GAP8, we have modified the *classification* example https://github.com/bitcraze/aideck-gap8-examples provided by *Bitcraze*. On the other hand, for embedding the algorithm for autonomous flight on the STM32, we have modified the app layer application *app_hello_world* of the *crazyflie* firmware https://github.com/bitcraze/crazyflie-firmware <br/>
A pre-trained quantized 8 bit model of *FlyDrosNet* is provided as TFlite model file ready for embedding.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/embedding_level.png" width=60% height=60%>

Finally, we prepared an arena with high contrast background for testing the drone. On the walls we placed a square, a rectangle, and a very big rectangle. According to the autonomous flight algorithm, the drone followed this sequence: take off -> go straight -> square detection -> turn left -> go straight -> rectangle detection -> turn right -> collision detection -> landing.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/autonomous_algorithm.png" width=40% height=40%>
<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/drone_test.gif">

- **data** folder contains the pattern dataset file with 3000 images for training and other with 300 for testing, labeled as (Collision, Rectangle, Square). It also contains the training results for each model.
- **images** folder contains the images used in this readme file.
- **models** folder contains the 3 models compared in this work each with a training framework, which generates the weights .h5 file and also the quantized TFlite model. It also generates the .mat files with the results of the training performance.
- **weights** folder contains the pre-trained weights of *FlyDrosNet* as .h5, and .tflite for the quantized version.
