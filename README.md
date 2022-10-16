# Insect-inspired-image-recognition-CNN
An image recognition Deep Learning model based on the visual system of fruit fly *Drosophila* for crazyflie drone deployment.

This is a CNN based on the visual system of the fly *Drosophila*. The network classifies images in 3 categories (Collision, Rectangle, Square), and it is though to be deployed in a crazyflie drone to perform an autonomous flight. A quantized 8 bit model is provided as TFlite model file.

- **data** folder contains a dataset file with 3000 images for training and other with 300 for testing.
- **images** folder contains the image with the workflow diagrams, and the training results.
- **models** folder contains the neural network with a training framework.
- **weights** folder contains the trained weights as TF model, and TFlite model.

<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/Fly_CNN_diagram.png" width=60% height=60%>
<img src="https://github.com/AngelCanelo/Insect-inspired-image-recognition-CNN/blob/main/images/Fly_CNN_performance.png">
