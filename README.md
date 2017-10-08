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

## Running the project:
### Requirements:
This submission requires a GPU enabled machine to run traffic light detection and classification.

A parameter is available to use ground truth data in the simulator instead of traffic light detection. Launch the `styx.launch` file with the following parameter.

`use_ground_truth:=true`

By default use_ground_truth is set to false.

On the first run, the traffic light detector will construct its trained graph file from file chunks stored in the project repository and save it as: `frozen_inference_graph.pb` for future use.

## Traffic Light detection

### Detection

Traffic light detection was accomplished by using the Tensorflow Object Detection API to build and train a model to recognize traffic lights in the full frame image from the vehicle's camera. The model was initially trained with the Bosch Small Traffic Light Dataset but it was found that this overfit the Bosch set and was not effective for the project. A pretrained standard model was implemented to detect traffic light locations and bounding boxes in the image and pass these to separate classifier. The model used for detection was RESNET101 with the COCO dataset

 - [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
 - [Bosch Small Traffic Light Dataset ](https://hci.iwr.uni-heidelberg.de/node/6132)

Note that when the model is initially loaded, the tensorflow graph file is constructed from mulitple chunks and then saved locally.

### Classfier

A classifier was implemented based on using the traffic light image from the detected bounding box and transforming to the HSV color space.  The image is then masked for Hue values in ranges for red, yellow and green. The red, yellow, green, white mask is then applied to the V: value channel of the image.  The V channel is then measured in the image locations corresponding to the red, yellow and green lights. The region with the highest intensity is used to determine the light state and the state is returned.


### Other approaches

Initially, we attemped to extract traffic light location in an image based on available poses for the car and traffic lights and using the CV2 pin hole camera projection function.  This proved infeasible as only the traffic light stop line location was given and the camera pose and focal length parameters were not given correctly. These could have been determined emperically, but at this point the Deep Learning based approach was acheiving good results. Other groups appeared to have some success with this geometrical approach to detemining traffic light locations.



## Waypoint updater

The concept of the waypoint updater is to use a jerk minimizing trajectory (JMT) to slow to a stop in front of red lights or accelerate to full speed on green. The largest challenge is that the JMT provides postion as a function of time rather than speed as a function of postion (or rather waypoint). After determining the JMT, time to reach each waypoint is determined by finding roots for the 5th order polynomial, excluding complex and physically impossible roots. The time derivative of the polynomial is taken to produce speed, and it is evaluated at the time for each waypoint.

## Drive by Wire Node

The DBW node uses a PID for speed control and the provided `YawController` for steering control. For speed control, the PID outputs are used after a low-pass filter, mainly to deal with noise in the measured current speed. Throttle commands are sent as percentages, and brake commands are send as torques.

The gains for the PID controller were tuned with the `twiddle` algorithm. A fake waypoint updater was created to feed the DBW node simple trapezoidal speed profiles (accelerate, constant speed, decelerate, rest; see `waypoint_updater/stop_go_profiler.py`) in a loop. In each iteration of the loop, the `SpeedPIDTuner` (`twist_controller/speed_pid_tuner.py`) measured the control error and also the smoothness of the control signal. Then, at the end of the iteration, it updated the gains for the next iteration using twiddle. Multiple runs with restarts arrived at similar gains, and this optimization technique was moderately effective at least up to a point --- the noise floor for the simulator's speed measurements was fairly high.

The `steer_ratio` for the simulator was wrong in the initial version of this sample code, which caused the car to understeer in the simulator. To try to find the right steer ratio, before Udacity provided one, several experiments were conducted in which the car was commanded to turn with a constant angle, and the radius of the resulting circle was measured. This showed the steering ratio for the simulator to be about 17.5, but when Udacity provided a value of 14.8, we used that instead. See `data/circle_plot.R` for details.

To check the sanity of the DBW node on Carla, the `dbw_test` node was fixed up (since Udacity did not provide the databag with which it was intended to be used) to use the `udacity_succesful_light_detection.bag`, with appropriate topic selections and remappings. The result was a new `max_throttle` parameter for the DBW node that is set to 1.0 in simulation and 0.025 in Carla. See `data/dbw_test.ipynb` for details.

## Testing

Testing was carried out prior to submission with the Udacity Simulator and the provided ROSbag data. Some discrepancies between the simulator and Carla were discovered such as required throttle settings.

All team members carrying out testing observed the car functioning well in the simulator and classifying traffic lights properly with the ROSbag data. Some minor corner cases in the simulator may still be untested.

The following video illustrates traffic light classification with the ROSbag data. 

<a href="http://www.youtube.com/watch?feature=player_embedded&v=cifYvqjr7ek
" target="_blank"><img src="http://img.youtube.com/vi/cifYvqjr7ek/0.jpg"
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>
