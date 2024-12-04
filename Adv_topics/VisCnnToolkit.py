####################################################################################
# VisCnnToolkit.py
#
# Author: Richard Dalzell
#
# This is a general CNN toolkit constructed to do the following
#
# 1.  Calculate and Display a GradCam heatmap from a input image and given Class
# 2.  Calculate and Display a Guided Backpropagation Image from a input image
# 3.  Display a set of feature-maps from a specified convolutional layer
# 4.  Display the trained filters from a specified convolutional layer
# 
# This tool currently supports two pre-trained CNN architectures;  VGG16 and Resnet50
# These networks have been trained on the ImageNet dataset and has 1000 classes
#     These classes are defined here; 
#       https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
#
# Usage: VisGradCam.py [-h] [--Image <filename>] [--ClassId <int>] 
#                      [--ConvLayer <string>] 
#                      [--Model Resnet50|VGG16] [--FeatureMap <bool>] [--Filters <int>] 
#                      [--Predict <bool>]
#                      [--ShowCams <bool>] [--Summary <bool>] 
#                      [--ShowRawHeatmap <bool>]
#
# options:
#   -h, --help                  Show this help message and exit
#   --Image <filename>          File Name of image to be explained
#   --ClassId <int>             Image Classification ID
#   --ConvLayer <string>        Name of Conv layer
#   --Model Resnet50|VGG16      ResNet50, VGG16
#   --FeatureMap True|False     Output Feature Map of ConvLayer
#   --Filters <int>             Display the pre-trained filters
#   --Predict True|False        Run a predication against the image
#   --ShowCams True|False       Show CAM generated images
#   --Summary True|False        Show Model Summary
#   --ShowRawHeatmap True|False Show the raw CAM image before scaling
#
#
# Some code snippets have been extracted or inspired from the following guides;
#    https://keras.io/examples/vision/grad_cam/
#    https://www.naukri.com/code360/library/guided-backpropagation
#    https://keras.io/examples/vision/visualizing_what_convnets_learn/
#
####################################################################################

import os, cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array


##########################################
# GradCAM heatmap class
# 
# Much of the code here has been adpated from this guide;
#   Code Reference: https://keras.io/examples/vision/grad_cam/
#
##########################################
class GradCAM:


    output_layer = None
    def __init__(self, model, layerName=None, outputLayerName=None):
        self.model = model
        self.layerName = layerName

        # Special case for certain CNN architectures 
        if outputLayerName:
            self.output_layer = self.model.get_layer(outputLayerName).output
        else:
            self.output_layer = self.model.output

    def compute_heatmap(self, image, classIdx, upsample_size, showRawHeatmap=False):

        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions        
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, 
                    self.output_layer]

        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            loss = preds[:, classIdx]
        grads = tape.gradient(loss, convOuts)
    
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = convOuts[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

        # Upscale the heatmap to the original image size so it can be overlayed
        scaled_heatmap = cv2.resize(heatmap.numpy(), upsample_size, cv2.INTER_LINEAR)

        # Optionally show the Raw Heatmap (pre and post scaling)
        if showRawHeatmap:
            _, subplot = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(6, 4))
            subplot[0,0].set_title('Raw heatmap')
            subplot[0,0].imshow(heatmap)
            subplot[0,1].set_title('Upsampled heatmap')
            subplot[0,1].imshow(scaled_heatmap)
            plt.tight_layout()
            plt.show()

        scaled_heatmap = np.expand_dims(scaled_heatmap, axis=2)
      
        return scaled_heatmap



##########################################
# GuidedBackprop class
#
# Code for this class was built around from the backprop guide;
#     Code Reference: https://www.naukri.com/code360/library/guided-backpropagation
#
#
# Guided Backpropagation is the combination of vanilla backpropagation at 
# ReLUs and DeconvNets. 
# ReLU is an activation function that deactivates the negative neurons. 
# DeconvNets are simply the deconvolution and unpooling layers. 
# We are only interested in knowing what image features the neuron detects. 
# So when propagating the gradient, we set all the negative gradients to 0. 
# We don’t care if a pixel “suppresses’’ (negative value) a neuron somewhere 
# along the part to our neuron. 
# Value in the filter map greater than zero signifies the pixel importance, 
# which is overlapped with the input image to show which pixel from the input 
# image contributed the most.
##########################################

# This is a custom activation function required by the backpropagation algorithm. 
#  Code Reference: https://www.naukri.com/code360/library/guided-backpropagation
@tf.custom_gradient
def guidedRelu(a):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(a>0, "float32") * dy
  return tf.nn.relu(a), grad

class GuidedBackprop:

    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.gbModel = Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )

        # Code Reference: https://www.naukri.com/code360/library/guided-backpropagation
        # Iterate through the model layers and build list of activation layers
        # Update the activation function to the custom activation function 'guidedRelu'
        layer_dict = [layer for layer in self.gbModel.layers[1:] 
            if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
    
    def guided_backprop(self, images, upsample_size):
        # Code Reference: https://www.naukri.com/code360/library/guided-backpropagation
        # We will use the Gradient tape to record the processed input image during 
        # the forward pass and calculate the gradients for the backward pass. 
        # Basically it is used to capture the gradients of the final
        # convolution layer.
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        # Generate image and up-scale it to the original input image size
        saliency = cv2.resize(np.asarray(grads), upsample_size)

        return saliency


##########################################
# This class creates a new model with a modified output layer to allow 
# visualisation of Feature Maps
# from the specified convulational layer.
# Useful utility for exploring how CNN work.
#
# Class is initialized using the model and the convulational layer 
# you wish to see visualisation
##########################################
class CnnExplorer:

    def __init__(self, model, layerName):
        # Update input model with a new output layer, 
        # this output layer is the conv layer 
        # that is to be visualized
        self.layerName = layerName
        self.model = model
        self.FeatureMapModel = Model(
            inputs=[model.inputs],
            outputs=model.get_layer(layerName).output
        )

    # This is the only class method, compute featuremap and display
    def display_featuremap(self, image, square=8):

        # This will return an array of FeatureMaps 
        # (based on the dimensions of the conv layer)
        FeatureMaps = self.FeatureMapModel.predict(image)
        
        # Display the feature maps - the number of maps displays will be square^2
        ix = 0
        _, subs = plt.subplots(nrows=square, ncols=square, squeeze=False, 
                        figsize=(10, 10))
        for i in range(square):
            for j in range(square):
                subs[i,j].axis('off')
                # plot feature map  in grayscale
                subs[i,j].imshow(FeatureMaps[0, :, :, ix])
                ix += 1

        # show the figure
        plt.show()

    # Display convulational filters 
    def display_filters(self, numFilters=6):

        # Retrieve the layer first
        layer = self.model.get_layer(self.layerName )

        # get filter weights
        filters, _ = layer.get_weights()

        # normalize filter values to 0-1 so we can visualize them
        f_min = filters.min()
        f_max = filters.max()
        filters = (filters - f_min) / (f_max - f_min)

        # Display first set of filters
        n_filters, ix = numFilters, 1
        _, subs = plt.subplots(nrows=n_filters, ncols=3, squeeze=False, 
                    figsize=(10, 10))

        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            # plot each channel separately  (this is assuming there 
            # the layer has 3 channels)
            for j in range(3):

                plt.imshow(f[:, :, j], cmap='gray')
                plt.show()
                subs[i,j].axis('off')
                # plot filter channel in grayscale
                subs[i,j].imshow(f[:, :, j], cmap='gray')
                ix += 1
        # show the figure
        plt.show()



#######################################################################################
############################ Utility Helper Functions here ############################
#######################################################################################

# Convert original based on requirement of CNN architecture - uses 
# tensorflow built-in methods
def preprocess(type, filename, target_size = (224,224)):
    img = img_to_array(load_img(filename, target_size = target_size))
    img = np.expand_dims(img, axis=0)
    if type == 'ResNet50':
        img = tf.keras.applications.resnet_v2.preprocess_input(img)
    else: # Default model is VGG16
        img = tf.keras.applications.vgg16.preprocess_input(img)
    
    return img


# Use model to make prediction with the pre-process image
# Display top 10 results ranked in order
def predict(type, model, processed_im):

    # Run model prediction
    preds = model.predict(processed_im)

    # Decode the predications based on model architecture
    if type == 'ResNet50':
        preds_list = tf.keras.applications.resnet_v2.decode_predictions(preds, top=10)
    else: # Default model is VGG16
        preds_list = tf.keras.applications.vgg16.decode_predictions(preds, top=10)

    rank = 1
    print ("***** TOP 10 PREDICTIONS *****")
    print ('{0:<10} {1:<15} {2:<8}'.format("rank", "name", "prob"))
    print ('{0:<10} {1:<15} {2:<8}'.format("----", "----", "----"))

    for p in preds_list[0]:
        (_, name, prob) = p
        print ('{0:<10} {1:<15} {2:<8}'.format(rank, name, prob))
        rank += 1


# Normalize image and convert into a RGB array
# Code Reference: https://keras.io/examples/vision/visualizing_what_convnets_learn/
def process_image(img):

    # normalize tensor: center on 0., ensure std is 0.25
    img = img.copy()
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.1

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")


    return img



# Image overlay Function 
def overlayGradCAM(img, heatmap):
    # Normalize heatmap to 0-255 unsigned ints (same as original input image)
    heatmap = np.uint8(255 * heatmap)
    # Use jet colourmap to colourize heatmap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Overlay image...adjusting image intensity ratio (0.3 for heatmap) 
    #           (0.5 for original image)
    new_img = 0.3 * heatmap + 0.5 * img

    return (new_img * 255.0 / new_img.max()).astype("uint8")


# Command Line Arg parsing
def  get_params():
    # Extract Command Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--Image', default='./test_images/goldfish.jpeg', 
            help='File Name of image to be explained')
    parser.add_argument('--ClassId', type=int, default=1, 
            help='Image Classification ID')  
    parser.add_argument('--ConvLayer', default='conv5_block3_3_conv', 
            help='Name of last Conv layer')
    parser.add_argument('--Model', default='ResNet50', help='ResNet50,  VGG16')  
    parser.add_argument('--FeatureMap', type=bool, default=False, 
            help='Output Feature Map of ConvLayer')
    parser.add_argument('--Filters', type=int, default=False, 
            help='Display the filters')  
    parser.add_argument('--Predict', type=bool, default=True, 
            help='Run a predication against the image')  
    parser.add_argument('--ShowCams', type=bool, default=True, 
            help='Show CAM generated images')  
    parser.add_argument('--Summary', type=bool, default=False, 
            help='Show Model Summary') 
    parser.add_argument('--ShowRawHeatmap', type=bool, default=False, 
            help='Show the raw CAM image before scaling') 


    args = parser.parse_args()
    return args

#######################################################################################



def showCAMs(originalImg, processImg, GradCAM, GuidedBP, ClassIdx, 
                upsampleSize, showRawHeatmap=False):

    # Compute Gradcam heatmap and upscale to original image size 
    cam = GradCAM.compute_heatmap(image=processImg, classIdx=ClassIdx, 
            upsample_size=upsampleSize, showRawHeatmap=showRawHeatmap)

    # Overlay on original iamge
    gradcam = overlayGradCAM(originalImg, cam)

    # Guided backprop
    gb = GuidedBP.guided_backprop(processImg, upsampleSize)
    gb_im = process_image(gb)

    # Guided GradCAM
    guided_gradcam = process_image(gb*cam)

    # Show four images;
    #     Original image (before preprocessing)
    #     Backpropagration saliceny map
    #     GradCAM heatmap overlay on original image
    #     GradCam overlay on BP salieny map
    _, subplot = plt.subplots(nrows=2, ncols=2, squeeze=False, figsize=(10, 10))

    subplot[0,0].set_title('Original image')
    subplot[0,0].imshow(cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB))
    subplot[0,0].axis('off')

    subplot[0,1].set_title('GradCam Heatmap')
    subplot[0,1].imshow(cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB))
    subplot[0,1].axis('off')

    subplot[1,0].set_title('Guided Backpropagation')
    subplot[1,0].imshow(cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB))
    subplot[1,0].axis('off')

    subplot[1,1].set_title('Masked Guided BP')
    subplot[1,1].imshow(cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB))
    subplot[1,1].axis('off')

    plt.tight_layout()

    plt.show()


#####################################################################
# Main - entry point
#####################################################################
if __name__ == "__main__":

    # Parse input parameters
    args = get_params()

    # Select CNN architecture, and load imagenet weights
    # Currently code only supports ResNet50 and VGG16
    # Potential update would be to accept any pre-trained CNN model.
    if args.Model == 'ResNet50':
        # Retrieve trained model (ResNet50), trained with 
        # ImageNet dataset (1000 classes)
        model = ResNet50V2(weights='imagenet')
    else: # Default to VGG16
        model = VGG16(weights='imagenet')

    if args.Summary:
        # Dump Model summary 
        model.summary()

    # Setup GBP class
    guidedBP = GuidedBackprop(model=model,layerName=args.ConvLayer)

    # Setup GradCam class
    gradCAM = GradCAM(model=model, layerName=args.ConvLayer)

    # Setup utility class for exploring CNN fearures
    # Currently this explorer class can extract featureMaps and features of the
    # specified ConvLayer
    CnnExplore = CnnExplorer(model=model, layerName=args.ConvLayer)

    # Read the input image
    originalImg = cv2.imread(args.Image)

    # Save the original image size, will be used to display the heatmap overlay
    upsampleSize = (originalImg.shape[1], originalImg.shape[0])
    
    # Read the input image and apply preprocessing according the the model type
    processedImg = preprocess(args.Model, args.Image)

    # If selected, display the feature maps of the supplied image using
    # Specified Conv Layer
    if args.FeatureMap:
        CnnExplore.display_featuremap(processedImg)

    # If selected, display the trained filters of the args.ConvLayer
    if args.Filters > 0:
        CnnExplore.display_filters(args.Filters)

    if args.Predict:
        # Run a standard prediction and display top 10 predictions 
        predict(args.Model, model, processedImg)

    if args.ShowCams:
        # Run the explainability code and show results
        showCAMs(originalImg, processedImg, gradCAM, guidedBP, 
                args.ClassId, upsampleSize, args.ShowRawHeatmap)