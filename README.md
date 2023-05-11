# Object-Detection-and-Tracking-using-UAV
## Project Description
An autonomous unmanned aerial vehicle (UAV) utilizes object detection (YOLOv7) and tracking (SORT) algorithms to effectively track and follow military tanks or any desired object.




https://github.com/SuneethJerri/Object-Detection-and-Tracking-using-UAV/assets/80540226/c6d9796e-dd22-4f8d-a27c-921150ba3c83



## Methodology 
- The UAV efficiently captures video using its onboard camera and transmits the footage to a computer using radio waves. The computer employs object detection to identify objects within the frame, and if any object is detected, it utilizes the SORT algorithm to track its movement and predict its path. Subsequently, the computer relays the object's relative coordinates to the UAV via telemetry, enabling the UAV to actively follow the object and ensure it remains within the frame.

- Our UAV, equipped with a PixHawk flight controller and Raspberry Pi 4 onboard, serves as a drone that will be utilized for controlling its operations.The Raspberry Pi 4 onboard the drone will be connected to the drone itself using a USB cable. Additionally, the drone will be equipped with a high-definition camera, specifically the Raspberry Pi Camera Module V2. The Raspberry Pi 4 then transmits the video footage to the ground station for running detections.

- Mission Planner was employed to conduct pre-flight checks, identify and resolve errors, and verify the availability and quality of GPS satellites prior to the drone's flight.

- Upon receiving the footage, the ground station executes an object detection algorithm (YOLOv7) to determine the presence of the desired object within the frame. To enhance detection accuracy, a rectangular frame is created using OpenCV to confine the object within a specific range of view. If the object is detected, the SORT algorithm is employed to track its movement.

- Given the known position of the UAV and the predicted position of the object relative to the UAV, the ground station sends the corresponding coordinates to the UAV. Consequently, the UAV adjusts its position to align the object relatively in the center of the frame, effectively maintaining its pursuit and ensuring continuous tracking.
### Ground Sample Distance
![90fBD](https://github.com/SuneethJerri/Object-Detection-and-Tracking-using-UAV/assets/80540226/886d3c97-f9ca-41bb-a23c-5073f367e900)

- The Ground Sample Distance (GSD) refers to the distance between pixels in an image. It provides an indication of the area covered by a single pixel based on the specific camera used and the altitude at which it is deployed.

- To calculate the GSD, we already have access to the necessary data, including the altitude, focal length, resolution, and sensor size of the camera employed. By utilizing the following formula: GSD = (Altitude x Focal Length) / (Resolution x Sensor Size), we can determine the GSD value.

- Utilizing the calculated GSD, we can further estimate the distance between the tanks and the drone, enabling the calculation of the resulting latitude and longitude coordinates of the tanks.
## Model 
### Training Data
- Due to the limited availability of drone footage or top-view shots of military tanks, it was challenging to gather a substantial number of relevant images for training the model. To overcome this obstacle, we collected whatever images we could salvage from existing drone footage available on the internet and even utilized images of a toy RC tank as substitutes. However, due to the scarcity of images, the model frequently suffered from overfitting, resulting in limited detection capabilities.

- Additionally, we explored the option of obtaining images from the game "World of Tanks" and "Fortnite". Nonetheless, the significant task of acquiring a sufficient quantity of game images and manually annotating them proved to be time-consuming, making it impractical within the given constraints.

- In order to expand our dataset, we made the decision to merge the tank images obtained from the internet with those captured from a toy RC tank, as well as including images from relevant games. Through data augmentation techniques, the combined dataset reached a total of approximately 1000 images

- Dataset link - https://universe.roboflow.com/aerial-detection/pure-tank
### Training
- Model training was done on a NVIDIA DGX-1.

- Since this is a single class detection problem, the model achieved an impressive average precision (AP) score exceeding 90%, indicating its excellent performance. Consequently, the model exhibits a high degree of accuracy in detecting tanks
## Object Detection and Tracking
### Why YOLOV7
- While attempting to train models using the available data, you encountered challenges with GPU utilization due to CUDA incompatibility. Consequently, you were unable to generate usable models when utilizing YOLOv3, YOLOv4, and YOLOv5.

- Fortunately, YOLOv7 proved to be compatible with your GPU, allowing for successful training. Not only did YOLOv7 deliver improved results compared to other models, but it also boasted the advantage of being the fastest real-time object detection algorithm available at that time. Given these factors, selecting YOLOv7 as the preferred choice for your project was a straightforward decision.
### SORT Algorithm
![03-architecture](https://github.com/SuneethJerri/Object-Detection-and-Tracking-using-UAV/assets/80540226/6e521145-344e-4a6b-bcaa-bfc6df63a2d2)

- Simple Online and Realtime Tracking or SORT algorithm is indeed a reliable and efficient algorithm for real-time tracking of multiple objects in video sequences. By utilizing a combination of Kalman filtering for state prediction and the Hungarian algorithm for association of detections with tracked objects, SORT provides accurate tracking results.

- In the context of your project, implementing SORT can enable the tracking of multiple tanks in a video. Each tracked tank will be assigned a unique ID, which can be used to specifically track a particular tank of interest. Although the implementation of tracking based on tank IDs has not been realized yet, it can be a potential development in the future, especially with the aid of a graphical user interface (GUI) to facilitate user interaction and selection.

- Implementing this feature would enhance the usability and flexibility of your system, allowing for targeted tracking and analysis of individual tanks within the video footage.
## Transmitting Data
### DroneKit
-DroneKit is a Python API that facilitates communication with the Pixhawk flight controller (PX4). It offers a high-level API that wraps around the ArduPilot API, leveraging the MAVLink protocol.

- For our project, we utilized the Pixhawk PX4 flight controller, which establishes communication with the onboard computer (such as a Raspberry Pi or Jetson).

- The onboard computer communicates with the ground station using the DroneKit API, utilizing telemetry functionality. The ground station can consist of a laptop or a mobile phone, provided it possesses sufficient processing power to receive the video stream from the camera and process it for tank detection purposes.

- Once the tanks are detected, and their latitude and longitude have been calculated, the ground station can transmit this information to the onboard computer using the DroneKit API via telemetry.
## Installation
- Clone the repository:
```python
git clone https://github.com/SuneethJerri/Object-Detection-and-Tracking-using-UAV
```
- Install the requirements:
```python
pip install -r requirements.txt
```
- Run the script:
```python
python detect_and_track.py --source 0 --weights weights.pt --baud 57600 --altitude 4 --connect com3 --view-img
```
   - Parameters
      - [source]: Specify the source for the script, which can be a video file or a webcam. Use 0, 1, or 2 as the source for the webcam (0 for default    webcam, 1 for the second webcam, and so on).
      - [weights_file_path]: Provide the path to the weights file for the detection and tracking model.
      - [baud_rate]: Set the baud rate of the Pixhawk flight controller.
      - [flight_altitude]: Specify the desired altitude at which the drone will fly.
      - [controller_port]: Specify the port to which the Pixhawk flight controller is connected (e.g., com3).
      - --view-img: Include this flag if you want to view the output video while the script is running.
      - Please ensure that you replace the placeholders in brackets ([]) with the appropriate values for your specific use case.
## Conclusion
We have successfully implemented the following functionalities in our project:

  - Real-time detection and tracking: Our system can detect and track tanks in real time, enabling continuous monitoring and analysis.

  - Geolocation transmission: The latitude and longitude coordinates of the detected tanks are efficiently sent to the drone, providing accurate positioning information.

  - Real-time tank tracking: The drone dynamically follows the tanks, ensuring consistent monitoring and capturing valuable data from different perspectives.

  - Fail-safe return: In the event of any failures or critical situations, the drone is programmed to autonomously return to the designated launch point, ensuring safety and mitigating risks.

These achievements demonstrate the effective integration of advanced technologies and algorithms in our project, enhancing the overall performance and reliability of our tank detection and tracking system.
## References and Articles
- YOLOV7 paper - https://arxiv.org/abs/2207.02696
- YOLOV7 - https://github.com/WongKinYiu/yolov7
- YOLOV7 with SORT - https://github.com/RizwanMunawar/yolov7-object-tracking
- Custom data training on YOLOV7 - https://medium.com/augmented-startups/yolov7-training-on-custom-data-b86d23e6623
- SORT paper - https://arxiv.org/abs/1602.00763
- DroneKit - https://dronekit.io/
- GSD - https://www.propelleraero.com/blog/ground-sample-distance-gsd-calculate-drone-data/#:~:text=GSD%20calculation%20considers%20the%20drone's,the%20altitude%20of%20the%20drone.
- GSD Calculator - https://www.propelleraero.com/gsd-calculator/
