###### This project is hosted on Github #####
https://github.com/schnwil/Image-Filter-Convolution-CUDA-OpenGL-OpenCV.git
######             


###
###Software Dependencies###
###
1. Latest cuda v8.0
2. Opencv version 2.4.3
3. Opencv libraries: opencv_core opencv_highgui -opencv_imgproc 


###
###Steps###
###
1. Change ARCH in makefile
2. make
3. ./matconv  

***Check key_bindings.pdf to try different kernels***

***NOTE***
1. Running above execuatable launches a GUI, so it has to be run on the machine with physical desktop connected. 
2. In order to run the code remotely, ssh X11 forwarding should be used which allows you to launch GUI using SSH tunneling remotely.
3. Project has been tested on Jetson TX1 board running Ubuntu 16.01

###
###log2csv###
###
Python script turns log.txt file into csv
1. Update path in log2csv.py file to path where log.txt is located
2. Run the python script
3. results.csv file in specified path created or updated
