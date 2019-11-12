# building-material-classification
Building material classification from Mapillary street-view images

## Introduction
In order to accurately predict the damage that a natural disaster would inflict on a building, detailed information on the building’s structure and composition material(s) is required. While all such information is virtually impossible to collect on a country level, since it would require complete, accessible and up-to-date cadaster records, an estimate of the building’s material can be obtained by simple visual inspection and is thus feasible to use.

While additional features of the building (e.g. foundation depth) will affect the likelihood of damage, we can assume that this latter can be factorized in the contribution of each building feature and that these contributions are to first order independent to each other. Working with such approximation, we can still give a meaningful approximation of the damage likelihood knowing exclusively the building material.

In this repo, we present a model to classify building material from street-level images of buildings, which enables quick extraction of this information on large areas. In particular, we used [Mapillary](https://www.mapillary.com/app/), a service for sharing crowdsourced geotagged photos, as a source of street-level images.

## Data
As a starting point, we chose an area of about 50 km2 in the south-east of Blantyre, southern Malawi, in which a total of 90 km of roads have been captured, corresponding to about 13700 street-level images. 

Mapillary employs computer vision models to detect specific objects in street-level images, among which buildings. For each image, these models provide:
* A list of objects detected in images
* The segment of the image corresponding to those objects
* A probability of the object being accurately detected 

We thus selected only the images in which one or more buildings were detected with high accuracy (>80%) and good resolution (>1% of the picture). 3233 images satisfied the selection criteria in the study area. From these images, we selected only the segments in which buildings were detected.

We then randomly chose 1000 images and have them labeled on Amazon Mechanical Turk, a crowdsourcing website for businesses. The labels referred to the buildings’ wall material and were chosen to be:
1. Concrete
2. Bricks
3. Corrugated metal sheet
4. Steel
5. Glass
6. Wood
7. Thatch / grass
8. Unclear / none of the above
Each image was labelled independently by 3 different people and we selected those images where 2 out of 3 labels agreed (889 out of 1000). We observe most of the buildings being labelled as either concrete or bricks. See this tab for the full results

| Label (Building Material) | Counts |
| ------------- | ------------- |
| Concrete  | 394  |
| Bricks  | 355  |
| Corrugated metal sheet | 25  |
| Steel  | 5  |
| Glass  | 11  |
| Wood  | 12  |
| Thatch / grass | 3  |
| Unclear / none of the above | 84  |

## Model

Given the results of the labelling task, we developed a model to classify a building as either made of bricks or concrete; we did not have sufficient data to include other materials. We chose to start with Convolutional Neural Networks (CNN), as such models represent the state-of-the-art in image classification. In particular, given the limited size of our training dataset, we used a pre-trained CNN, [ResNet50](https://arxiv.org/abs/1512.03385), to extract high-level features from the images, and trained a 2-layer fully-connected network on top of it. The parameters of the underlying ResNet50 model were kept fixed, i.e. the model was not re-trained.
We used the [keras framework](https://keras.io/) from image pre-processing and model building/training. In particular, we used data augmentation routines to extend the training dataset, by applying rotations, reflections and translations to the original images. We chose Adam as optimization algorithm, with a fixed learning rate of 1e-3, and we trained the model with mini-batches of 8 images each. The model reached a top accuracy (recall) of about 85% () after about 50 epochs.





