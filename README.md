# MARS CARS

Udacity Term 3 Capstone Project

## Team Members:


| Name        | Email           | Slack  |
| ------------- |:-------------:| -----:|
| Brendan Schell     | brendanschell1@gmail.com | @schellbrendan |
| Miguel A. Roman      | mangelroman@gmail.com      |   @mangelroman |
| John Lees-Miller | jdleesmiller@gmail.com     |    @jdleesmiller |
| Kevin McFall | kmcfall@kennesaw.edu     |    @kmacprof |
| Robbie Edwards | rob.a.edwards@gmail.com     |    @waterfox |


## Traffic Light detection

### Detection

Traffic light detection was accomplished by using the tensorflow object detection API to build and train a model to recognize traffic lights in the full frame image from the vehicle's camera. The model was initially trained with the Bosch Small Traffic Light Dataset but it was found that this overfit the Bosch set and was not effective for the project. A pretrained standard model was implemented to detect traffic light locations and bounding boxes in the image and pass these to separate classifier. The model used for detection was Resnet101 with the COCO dataset

 - [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
 - [Bosch Small Traffic Light Dataset ](https://hci.iwr.uni-heidelberg.de/node/6132)

Note that when the model is initially loaded, the tensorflow graph file is constructed from mulitple chunks and then saved locally.

### Classfier

A classifier was implemented based on using the traffic light image from the detected bounding box and transforming to the HSV color space to check for Hue values in ranges for red, yellow and green to classify the lights. The V: value parameter was also measured in the image locations corresponding to the red, yellow and green lights. The information extracted form the H and V channels was combined and scored and used to classify the traffic light state.


### Other approaches

Initially, we attemped to extract traffic light location in an image based on available poses for the car and traffic lights and using the CV2 pin hole camera projection function.  This proved infeasible as only the traffic light stop line location was given and the camera pose and focal length parameters were not given correctly. These could have been determined emperically, but at this point the Deep Learning based approach was acheiving good results. Other groups appeared to have some success with this geometrical approach to detemining traffic light locations.



## Waypoint updater

The basic concept is to use a jerk minimizing trajectory (JMT) to slow to a stop in front of red lights or accelerate to full speed on green. The largest challenge is that the JMT provides postion as a function of time rather than speed as a function of postion (or rather waypoint). After determining the JMT, time to reach each waypoint is determined by finding roots for the 5th order polynomial, excluding complex and physically impossible roots. The time derivative of the polynomial is taken to produce speed, and it is evaluated at the time for each waypoint.

## Drive by Wire Node
