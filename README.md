Alphanumeric Character Classifier

This project uses a subset of the HASYv2 dataset which can be found here: https://github.com/sumit-kothari/AlphaNum-HASYv2

**model_config.py:**

Used to compare two different network architectures to determine which one performs better, runs a specified number of trials as follows:
1. Construct both models
2. Train and validate both models on the same training and validation set for the specified number of epochs (training and validation accuracy vs epochs can be graphed for both models if the corresponding lines in model_config.py are uncommented)
3. Run both models on the same test set
4. Record whether or not the trial model had a higher median/mean/max validation accuracy or test accuracy than the baseline model
After all trials have completed, output how many times the trial model beat the baseline model on the aforementioned statistics

**classifier.py**

This file is the completed project. It works as follows:
1. Train a sequential Keras model (network architecture was decided upon after many different configurations were tested in *model_config.py*) on all data present
2. Create a 640 x 640 Turtle Graphics window that has been confugured to allow the user to click and drag the turtle around to draw an alphanumeric character
3. When the user presses the "s" key on their keyboard after they have finished drawing, the drawing is scaled down to a 32 x 32 image in PostScript format
4. Show the scaled-down version of the drawing to the user
5. Reshape the scaled-down image to match the required input size for the network
6. Run the network on the image and print out the prediction
