1. Data generation:
   2. Generate domain 1 imbalanced data with [imabalanced_data_generation.ipynb](data_generation%2Fimabalanced_data_generation.ipynb)
   3. Run [test_data_generation.ipynb](data_generation%2Ftest_data_generation.ipynb) to generate test data with distribution shift
   3. Save labels and images as two different files for easy loading in next steps.

2. Run [modularEyeTrain.py](modularEyeTrain.py) with the training dataset and save its weights.
   3. Check the joint distribution it is matching.
3. Run [generateIntvData.py](generateIntvData.py) to generate images from style-gan with label outputs from the just trained model.
   * Count each combination of Eyeglasses and Male in the interventional samples.
   * Generate images according to the occurrence with InterfaceGAN.
4. Train Classifiers
   5. Train [traindomain_classifier.py](TrainEyeglassClassifiers%2Ftraindomain_classifier.py) on the training dataset
   6. Train [intv_classifier.py](TrainEyeglassClassifiers%2Fintv_classifier.py) on stylegan generated intv dataset
   7. Train [augmented_classifier.py](TrainEyeglassClassifiers%2Faugmented_classifier.py) on the combined dataset.
7. Run [compare_all_classifiers.py](TrainEyeglassClassifiers%2Fcompare_all_classifiers.py) to test performance of
all classifiers on test domain data in different sub-population.
8. Run [compare_results.ipynb](TrainEyeglassClassifiers%2Fcompare_results.ipynb) to draw plot with the results.