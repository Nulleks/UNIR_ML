# Person detection - Faster rcnn inception v2

Trained with [Tensorflow](https://github.com/tensorflow/models/tree/master/research/object_detection)

Dataset: [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

Download model: https://1drv.ms/f/s!AorkmCSibVijg20DGhsGaRYvTDi5 (Put in the root folder of the repository)


## Usage
`python predict.py -i /path/to/image/or/video or webcam -o /path/output/directory(optional)`

Example:

`python predict.py -i sample.jpg`

`python predict.py -i webcam`
