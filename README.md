# Sushi or Sandwich classifier

I present 2 classifiers implemented in Keras using Tensorflow backend in the [Jupyter notebook](sushi-or-sandwich-keras.ipynb):

- First attempt, a Convolutional Neural Network model,
- Second attempt, a transfer learning model trained using features extracted from MobileNet (ImageNet weights).

To train the models, I split the images into train-test datasets using 80%:20% (642:162) ratio. Since there are only 642 training images, I also implement a data augmentation function where the input images are randomly flipped horizontally, sheared, zoomed and rotated.

## Convolutional Neural Network model

This model has 3 convolution layers with max-pooling and 2 dense layers at the top to predict feature labels. All activation functions between the convolution and the dense layers are `relu` except the probabilities after the top layer are treated with a`sigmoid` activation function as this normalises the values to a range between 0 and 1 making it suitable for binary classification problems like this one. A drop-out probability of 0.5 is applied during the training phase between the fully connected layers. The model uses `rmsprop` optimiser to minimise binary cross entropy.

The model achieves an accuracy of ~70% and an AUC of ~0.78 on the test dataset after 39 epochs (early stopping after no accuracy improvements for 20 epochs).

A summary of the CNN model is as follows:

	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 128, 128, 3)       0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 126, 126, 32)      896       
	_________________________________________________________________
	activation_1 (Activation)    (None, 126, 126, 32)      0         
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 63, 63, 32)        0         
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 61, 61, 32)        9248      
	_________________________________________________________________
	activation_2 (Activation)    (None, 61, 61, 32)        0         
	_________________________________________________________________
	max_pooling2d_2 (MaxPooling2 (None, 30, 30, 32)        0         
	_________________________________________________________________
	conv2d_3 (Conv2D)            (None, 28, 28, 64)        18496     
	_________________________________________________________________
	activation_3 (Activation)    (None, 28, 28, 64)        0         
	_________________________________________________________________
	max_pooling2d_3 (MaxPooling2 (None, 14, 14, 64)        0         
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 12544)             0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 64)                802880    
	_________________________________________________________________
	activation_4 (Activation)    (None, 64)                0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 64)                0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 1)                 65        
	_________________________________________________________________
	activation_5 (Activation)    (None, 1)                 0         
	=================================================================
	Total params: 831,585
	Trainable params: 831,585
	Non-trainable params: 0
	_________________________________________________________________

## Transfer learning model trained using features extracted from MobileNet (ImageNet weights)

MobileNet is a light-weight neural network model engineered by Google. The weight files are around ~16MB compared to ~548MB for VGG16 network which is proportional to the number of parameters in the model making MobileNet ideal for mobile vision applications.

To improve the previous CNN model, features are extracted from MobileNet (using ImageNet training weights). The features are then used to train 2 fully connected layers to predict feature labels. A `relu` activation function is used between the fully connected layers and the final probabilities are treated with a `sigmoid` activation function. A drop-out value of 0.5 is applied during the training phase between the fully connected layers. Like before, the model uses `rmsprop` optimiser to minimise binary cross entropy. 

The model achieves an accuracy of ~85% and an AUC of ~0.93 on the test dataset after 49 epochs (early stopping after no accuracy improvements for 20 epochs). This is a clearly an improvement over the previous model both in terms of accuracy and AUC.

A summary of the top layer stacked upon MobileNet is as follows:

	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	flatten_2 (Flatten)          (None, 16384)             0         
	_________________________________________________________________
	dense_3 (Dense)              (None, 128)               2097280   
	_________________________________________________________________
	activation_6 (Activation)    (None, 128)               0         
	_________________________________________________________________
	dropout_2 (Dropout)          (None, 128)               0         
	_________________________________________________________________
	dense_4 (Dense)              (None, 1)                 129       
	_________________________________________________________________
	activation_7 (Activation)    (None, 1)                 0         
	=================================================================
	Total params: 2,097,409
	Trainable params: 2,097,409
	Non-trainable params: 0
	_________________________________________________________________

## Suggested improvements

- Gather more training data
- Implement additional data augmentation strategies
- Harness the power of GPU to facilitate training on larger networks

In production, a model like this that has been extended to detect additional categories of user uploaded images could be used to annotate them which could be used to learn the types of meals a user likes to eat and share with others in order to recommend recipes shared by others to the user. To this end, in order to deploy this model as a useful product, it would need to be trained on additional categories of meals. Since food categories are different compared to ImageNet categories, it may be necessary to train all of the earlier layers of MobileNet to improve overall classification accuracy.

## To build the container using Dockerfile:

	WORKDIR=~/Coding/SushiSandwichClassifier
	cd $WORKDIR
	docker build -t sushi .

## To run a Jupyter notebook server from the container:

	WORKDIR=~/Coding/SushiSandwichClassifier
	docker run -it -p 9999:9999 -v $WORKDIR/:/opt/sushi/ sushi jupyter notebook --port 9999 --ip=0.0.0.0 --allow-root --no-browser --notebook-dir=/opt/sushi/

Navigate to http://localhost:9999/?token=[TOKEN]