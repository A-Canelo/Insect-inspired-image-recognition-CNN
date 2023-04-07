# Insect-inspired-image-recognition-CNN
An image recognition Deep Learning model based on the visual system of fruit fly *Drosophila*, *FlyVisNet*, for embedding on a *crazyflie 2.1* drone STM32 and *ai-deck* GAP8 to perform an autonomous flight.

## Architecture design
*FlyVisNet* is a CNN based on the visual system of the fly *Drosophila*. The architecture uses the neural pathways necessary for feature and looming detection. It has 3 outputs to classify images in 3 categories (Collision, Rectangle, Square).

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/FlyVisNet_diagram.png" width=70% height=70%>

We also provide a pattern dataset labeled as mentioned, and compared the performance results to other relevant architectures, showing that we have achived a sufficient accuracy using low memory, which is essential for embedded deployment.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/pattern_dataset_sample.png" width=35% height=35%>

| Architecture  | Top accuracy (%) | Parameters (#) | Memory (KB) |
| :---: | :---: | :---: | :---: |
| ResNet101  | 97.66  | 42,658,051 | 489,290 |
| MobileNetV2  | 96.66  | 2,261,251 | 26,450 |
| FlyVisNet  | **95.33**  | 747,665 | **8,968** |
| FlyVisNet_8bit  | **84.00**  | 747,665 | **753** |

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/performance_comparison.png" width=40% height=40%>

For embedding *FlyVisNet* on the *ai-deck* GAP8, we have modified the *classification* example https://github.com/bitcraze/aideck-gap8-examples provided by *Bitcraze*. On the other hand, for embedding the algorithm for autonomous flight on the STM32, we have modified the app layer application *app_hello_world* of the *crazyflie* firmware https://github.com/bitcraze/crazyflie-firmware <br/>
A pre-trained quantized 8 bit model of *FlyVisNet* is provided as TFlite model file ready for embedding.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/End-to-end_framework.png" width=65% height=65%>

Finally, we prepared an arena with high contrast background for testing the drone. On the walls we placed a square, a rectangle, and a very big circle. According to the autonomous flight algorithm, the drone followed this sequence: take off -> go straight -> square detection -> turn left -> go straight -> rectangle detection -> turn right -> collision detection -> landing.  
In a second test the drone performed a surveillance flight according to the second algorithm. The drone followed this sequence: take off -> go straight -> rectangle detection -> go straight -> collision detection -> turn away -> go straight -> rectangle detection -> go straight -> collision detection -> turn away -> go straight -> rectangle detection -> go straight -> turn away -> landing.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/autonomous_flight_algorithm.png" width=50% height=50%>
<p float="left">
  <img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/visually-guided.gif" />
  <img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/surveillance.gif" />
</p>
https://youtu.be/Tedu2W9-55s
</p>
https://youtu.be/LNgJ-dkv7S0
</p>
---
## Deployment
The necessary components for deployment are as follow:
- Crazyflie 2.1 drone
- Crazyradio PA 2.4 GHz USB dongle
- Flow deck v2
- AI deck 1.1

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/necessary_components.jpg" width=40% height=40%>

Instructions for deployment on *crazyflie 2.1* and *ai-deck*:
- Download *bitcraze-vm* https://github.com/bitcraze/bitcraze-vm/releases
- On the vm clone *aideck-gap8-examples*, and *crazyflie-firmware* repositories: <br/>
https://github.com/bitcraze/aideck-gap8-examples <br/>
https://github.com/bitcraze/crazyflie-firmware
- Substitute the folder *classification* in `aideck-gap8-examples/examples/ai/` by the provided by us in `deployment/classification`
- Substitute the folder *app_hello_world* in `crazyflie-firmware/examples/` by the provided by us in `deployment/app_hello_world`

- Build and flash on *ai-deck* GAP8. In folder `aideck-gap8-examples`:
```
$ docker run --rm -v ${PWD}:/module aideck-with-autotiler tools/build/make-example examples/ai/classification clean model build image
```
```
$ cfloader flash examples/ai/classification/BUILD/GAP8_V2/GCC_RISCV_FREERTOS/target.board.devices.flash.img deck-bcAI:gap8-fw -w radio://0/80/2M/E7E7E7E7E7
```
- Build and flash on *crazyflie* STM32. In folder `crazyflie-firmware/examples/app_hello_world`:
```
$ make all clean
```
```
$ cfloader flash ./build/cf2.bin stm32-fw -w radio://0/80/2M/E7E7E7E7E7
```

Folders:
- **data** folder contains the pattern dataset file with 3000 images for training and other with 300 for testing, labeled as (Collision, Rectangle, Square). It also contains the training results for each model.
- **deployment** folder contains the codes for the deployment of the *FlyVisNet* on *ai-deck* GAP8, and autonomous flight algorithm on STM32.
- **images** folder contains the images used in this readme file.
- **models** folder contains the 3 models compared in this work each with a training framework, which generates the weights .h5 file and also the quantized TFlite model. It also generates the .mat files with the results of the training performance. The file *performance_comparison.py* plots the results.
- **weights** folder contains the pre-trained weights of *FlyVisNet* as .h5, and .tflite file for the quantized version.
