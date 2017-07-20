implementation of phy layer using CUDA on nvidia GPU

Notes:
Tx_Generic and Tx_Generic_Streams are the final codes for the Tx.
Receiver is not implemented as a generic code. You'll find each chain folder independantly (SISO,2x2 MIMO, etc...)


Timing should be measured with the functions in the file (main.cuh) in (Tx_Generic or Tx_Generic_Streams) folders.
This function is the correct one (it uses the library chrono.h)
Other folders may use (cudaEvent_t) for measuring time which doesn't count CPU time.

To test the output:
We have written a section at the end of main function that creates a (.m) file called (output.m) and another one called (matlab_test).
You should run output.m in MATLAB and it'll print the summation of the error between the 2 implementations (CUDA and MATLAB)

Important Note:
You'll face some error when you use CUDA with Visual Studio
1) math_helper.h is not found:
move the files inside this folder (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA Samples\v8.0\common\inc) or (C:\ProgramData\NVIDIA GPU Computing Toolkit\CUDA Samples\v8.0\common\inc) (depending on where you installed CUDA_Samples) to this folder (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include)

2) You have to add some static libraries. Go to project properties in VS --> Linker --> Input --> append this line to additional dependencies (cufft.lib;cudart.lib;)

3) To add the input files: Go to project properties in VS --> Debugging --> add input_files to Command arguments:
e.g. D:\input_1x1.txt D:\ri_0.txt
