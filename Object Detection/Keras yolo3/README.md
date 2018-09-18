# Person detection - YOLOv3

Trained with [Keras-YOLO3](https://github.com/experiencor/keras-yolo3)

Dataset: [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/)

Download model: uploading


## Usage
`python predict.py -c config.json -i /path/to/image/or/video or webcam -o /path/output/directory(optional)`

Example:

`python predict.py -c config.json -i sample.jpg`

`python predict.py -c config.json -i webcam`
