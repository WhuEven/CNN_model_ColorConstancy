# Color Constancy(CC) using CNN

Solving the color constancy porblem with CNN model proposed by [Simone_et_al](https://arxiv.org/pdf/1504.04548.pdf)

## Getting Started


### Prerequisites

* You will need OpenCV library, Tensorflow and Keras in order to run the code.

* Also, the dataset for CC problem I used here is of Shi and Gehler you can find more information and download link [here](http://www.cs.sfu.ca/~colour/data/shi_gehler/).

* [Here](https://drive.google.com/file/d/1aHx7v-VGREQfyemsGqB8-1Ccfz4lrzKr/view?usp=sharing) is the smaller set from Shi-Gehler for you to use with the CNN model. 

* If you prefer a fastest way to play with the model, you can skip directly to the part **"Running the tests"** where you can use my pre-trained CNN model to see how it works against the CC problem. 

### Establishing Dataset

* The idea of the algorithm is training the CNN on the patches sampled from the image. To save you sometime I've done the job for you and you can download the sampling patches [here](https://drive.google.com/file/d/1kr_EHp5we6AkzdpKLCbeYhTxfMJPhceO/view?usp=sharing) ~ 4GB
and put it inside your current working folder, like this:

```
.../CNN_model_ColorConstancy/patches
```

* And then, you can create your own train and test set by running the script 

```
generate_data.py
```
* You can change the size of your train(test) set and number of train(test) ground-truth illuminants by simply changing these arguments:

```
train_size, test_size, number_of_train_gt, number of_test_gt 
```

* However, if you prefer a faster way, you can directly download the train and test set I have created [here](https://drive.google.com/file/d/1w-qfkDugvs1oUdob2_DI4uo26OuGUK-S/view?usp=sharing).
These are .npy file, you can simply use **numpy.load** to load them:

```
X_train = np.load('X_train.npy');...
```

### Traing the model

After finish preparing the dataset, you can start training the model with **CNN_keras.py**, simply load your train and test set and run it.

**Remarks :** This is the problem of CC so i used the loss function of **"cosine_proximity"** as it is closest to the **"[angular error](https://fr.mathworks.com/help/images/examples/comparison-of-auto-white-balance-algorithms.html)"**.


## Running the tests

* You can find my pre-trained models and weights in these file:

```
cc_model.h5 <---weights
cc_model.json <--pre-trained model
```

* If you wish to start your own model, simply delete it and start from **"Establishing Dataset"** to build your own. However your choice, you can test the models with the images download from the smaller data set of Shi-Gehler I've provided above (**Prerequisites**).

* Now, you can easily test the model by running the **white_balancing.py**, for example: 

```
img = cv2.imread('0001.png')
image_name = '0001'
patch_size = (32, 32)
img_white_balance = white_balancing(img, image_name, patch_size)
```

Remark: modify the argument **image_name** to the name of the image you want to test in the **Color-casted** folder:
(remember to put the image inside the same folder of your test_illum.py, otherwise you have to include the path in the **image_name** argument)

```
image_name = '0005' 
```

if not at the same folder:

```
image_name = '.../yourfolder/0005' 
```
* After running the script, you can compare the results with the corresponding image in the **Ground-truth** folder.

### Some notes

* There are many things hard to explain if you are not familiar with color science. For this reason, if you have questions, do not hesitate to contact me.


## Authors

* **Simone Bianco** - *Initial work* - [Color Constancy Using CNNs](https://arxiv.org/pdf/1504.04548.pdf)

* **Hien Pham** - *Re-implementation*

## License

This project is under license of Technicolor.

## Acknowledgments

* This is the implementation of Simone Bianco works. If you use this code for research purposes please cite [Simone's work](https://arxiv.org/pdf/1504.04548.pdf) and my implementation in the references.



