# RESLAM: A real-time robust edge-based SLAM system

**Please note that RESLAM is a research project and its code is released without any warranty. RESLAM will most likely not be developed any further**

In this work, we present **RESLAM**, a robust edge-based SLAM system for RGBD sensors. Edges are more stable under varying lighting conditions than raw intensity values, which leads to higher accuracy and robustness in scenes, where feature- or photoconsistency-based approaches often fail. The results show that our method performs best in terms of trajectory accuracy for most of the sequences indicating that edges are suitable for a multitude of scenes.

## If you use this work, please cite any of the following publications:
* **RESLAM: A real-time robust edge-based SLAM system**, Schenk Fabian, Fraundorfer Friedrich, ICRA 2019, [pdf](https://github.com/fabianschenk/fabianschenk.github.io/raw/master/files/schenk_icra_2019.pdf)
* **Combining Edge Images and Depth Maps for Robust Visual Odometry**, Schenk Fabian, Fraundorfer Friedrich, BMVC 2017, [pdf](https://github.com/fabianschenk/fabianschenk.github.io/raw/master/files/schenk_bmvc_2018.pdf),[video](https://youtu.be/uj3rRyqSEnQ)
* **Robust Edge-based Visual Odometry using Machine-Learned Edges**, Schenk Fabian, Fraundorfer Friedrich, IROS 2017, [pdf](https://github.com/fabianschenk/fabianschenk.github.io/raw/master/files/schenk_iros_2017.pdf), [video](https://youtu.be/PUTV9vsdpbA)

## License
RESLAM is licensed under the [GNU General Public License Version 3 (GPLv3)](http://www.gnu.org/licenses/gpl.html).

If you want to use this software commercially, please contact us.

## Building the framework
So far, the framework has only been built and tested on the following system.
### Requirements
* [Ubuntu 16.04, 15.10, 17.04](https://www.ubuntu.com/)
* [OpenCV > 3](http://opencv.org/)
* [Eigen > 3.3](http://eigen.tuxfamily.org/index.php?title=Main_Page)
* [Ceres >= 1.13](http://ceres-solver.org/installation.html)


[Sophus](https://github.com/strasdat/Sophus) is now part of this repository (in thirdparty/Sophus).

Building on Windows and backwards compatibility might be added in the future.

### Optional
Set the optional packages in the cmake-gui
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)  (for graphical viewer)


### Build commands
```bash
git clone https://github.com/fabianschenk/RESLAM
cd RESLAM
mkdir build
cd build
cmake . ..
make -j
```

### Known Issues
#### Segmentation Fault with Ceres/Eigen [#2](https://github.com/fabianschenk/RESLAM/issues/2), and [#3](https://github.com/fabianschenk/RESLAM/issues/3).
Some people report a problem with Ceres/Eigen.
Please, have a look at [#2](https://github.com/fabianschenk/RESLAM/issues/2), and [#3](https://github.com/fabianschenk/RESLAM/issues/3).
Make sure that you have the latest (stable) [Eigen version 3.3.X](http://eigen.tuxfamily.org/index.php?title=Main_Page) and that it matches the one used by Ceres.

#### Segmentation fault after repeated tracking losses [#3](https://github.com/fabianschenk/RESLAM/issues/3)
In some sequences, e.g. `freiburg2_large_with_loop`, there are depth maps containing mostly invalid values.
The problem is that the Kinect and most other RGBD sensors cannot reconstruct surfaces far away from sensor (around > 6 m) due to the small baseline of the sensor.
In such cases, RESLAM does not work and might fail with a segmentation fault after repeated tracking losses. This issue will hopefully be fixed in the future.


## How to reproduce the results from the paper

**If you enable multi-threading, results might differ a bit since float additions are not executed in the same order during each run!**

### [TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)
Download the sequence you want to test and specify the "associate.txt" file in the dataset_tumX.yaml settings file.

To generate an "associate.txt" file, first download the "associate.py" script from [TUM RGBD Tools](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/) and then run
```bash
python associate.py DATASET_XXX/rgb.txt DATASET_XXX/depth.txt > associate.txt
```
in the folder, where your dataset is.
 
In the "RESLAM" directory:
```bash
build/RESLAM config_files/reslam_settings.yaml config_files/dataset_tum1.yaml
```
For evaluation of the absolute trajectory error (ATE) and relative pose error (RPE) download the corresponding scripts from [TUM RGBD Tools](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/).

 
## Supported Sensors

Support for other sensors such as Orbbec Astra Pro and Intel RealSense can be adapted from [REVO](https://github.com/fabianschenk/REVO).

<!--- REVO supports three different sensors at the moment:
* [Orbbec Astra Pro Sensor](https://orbbec3d.com/product-astra-pro/)
* [Orbbec Astra Sensor](https://orbbec3d.com/product-astra/)
* [Intel Realsense ZR300 (other versions are untested!)](https://click.intel.com/intelr-realsensetm-development-kit-featuring-the-zr300.html)

For the Intel sensor set "WITH_REALSENSE", for the Orbbec Astra Pro set "WITH_ORBBEC_FFMPEG" (recommended) or "WITH_ORBBEC_UVC" (not recommended, requires third party tools) and for the non-pro Orbbec Astra set "WITH_ORBBEC_OPENNI"!
**Note:** Make sure that you set the USB rules in a way that the sensor is accessible for every user (default is root only).

REVO can be compiled for all three sensors only if WITH_REALSENSE, WITH_ORBBEC_FFMPEG and WITH_ORBBEC_OPENNI are set.
If WITH_ORRBEC_UVC is set, there is a conflict with the librealsense!
To solve this issue, use WITH_ORBBEC_FFMPEG!

The sensor to be used is determined from the INPUT_TYPE set in the second config file.
For Orbbec Astra Pro INPUT_TYPE: 1, for Intel Realsense INPUT_TYPE: 2 and for Orbbec Astra INPUT_TYPE: 3.

Example config files for all three sensors can be found in the config directory!
### Intel RealSense ZR300
Install [librealsense](https://github.com/IntelRealSense/librealsense), set the intrinsic parameters in the config file.
This framework was tested with the Intel RealSense ZR300.

### Orbbec Astra Sensor
The (non-pro) Orbbec Astra Sensor can be fully accessed by Orbbec's OpenNI driver.
First [download the openni driver](https://orbbec3d.com/develop/#registergestoos) and choose the correct *.zip file that matches your architecture, e.g. OpenNI-Linux_x64-2.3.zip. 
Extract it and copy libOpenNI2.so and the "Include" and "OpenNI2" folder to REVO_FOLDER/orbbec_astra_pro/drivers. 

### Orbbec Astra Pro Sensor
#### With FFMPEG
The standard OpenNI driver can only access the depth stream of the [Orbbec Astra Pro Sensor](https://orbbec3d.com/product-astra-pro/), thus we have to access the color stream via FFMPEG.
Install the newest FFMPEG version
```bash
sudo apt install ffmpeg
```
or download from [FFMPEG Github](https://www.ffmpeg.org/download.html).
#### With LibUVC (not recommended)
The standard OpenNI driver can only access the depth stream of the [Orbbec Astra Pro Sensor](https://orbbec3d.com/product-astra-pro/), thus we have to access the color stream like a common webcam.
*Note: We use libuvc because the standard webcam interface of [OpenCV](http://opencv.org/) buffers the images and doesn't always return the newest image.*

First [download the openni driver](https://orbbec3d.com/develop/#registergestoos) and choose the correct *.zip file that matches your architecture, e.g. OpenNI-Linux_x64-2.3.zip. 
Extract it and copy libOpenNI2.so and the "Include" and "OpenNI2" folder to REVO_FOLDER/orbbec_astra_pro/drivers. 

Then install [Olaf Kaehler's fork of libuvc](https://github.com/olafkaehler/libuvc) by performing the following steps in the main directory.
```bash
cd ThirdParty
git clone https://github.com/olafkaehler/libuvc
cd libuvc
mkdir build
cd build
cmake . ..
make -j
make install
```
## Troubleshooting
### Sophus
There was a problem with the old REVO version and a new Sophus version that introduced orthogonality checks for rotation matrices. 
If you face such an error, simply check out the current version of REVO.
### Orbbec with LIBUVC and Intel Realsense
If WITH_ORRBEC_UVC is set, there is a conflict with the librealsense! To solve this issue, use WITH_ORBBEC_FFMPEG!-->
