# kneeoa
Progression of Knee OsteoArthritis

Steps to follow:

1) Download Data- .
  Download images from server and store in folder eg. imagedata. Download progressors.csv and non-progressors.csv
  Download model vgg16.tfmodel
  Run the parseaandstore.py to generate train_file and test_file which contains labels

  Run the splitter.py which splits the imagedata into train image and test image folder based on labels file 

Training images folder - All images for training
Testing images Folder - All images for testing
Training image labels file - Pickled file with training labels
Testing image labels file - Pickled file with testing labels

2) Extract features(CNN Codes) from the maxpool:5 layer of PreTrained CovNet(VggNet) and save them beforehand for faster training of Neural network.

python train.py <Training images folder> <Testing image folder> <Train images codes folder > <Test images codes folder>

Train images codes folder - Path where training images codes will be stored
Test images codes folder - Path where testing images codes will be stored

3) The extracted features are now used for training our 2-Layer Neural Network from scratch.The computed models are saved as tensorflow checkpoint after every Epoch.
python train_model.py final_train_images_calc_nodule_only train_image_codes ../parser/train_file savedmodel

4) Finally the saved models are used for making predictions.Confusion Matrix is used as the Performance Metrics for this classifcation task.

python test_model.py final_test_images_calc_nodule_only test_image_codes ../parser/test_file savedmodel
