%% Fog Rectification 
% 
% This example shows how to remove fog from images captured under foggy 
% conditions. The algorithm is suitable for FPGAs.
%
% Fog rectification is an important preprocessing step for applications in
% autonomous driving and object recognition. Images captured in foggy and
% hazy conditions have low visibility and poor contrast. These conditions
% can lead to poor performance of vision algorithms performed on foggy images.
% Fog rectification improves the quality of the input images to such algorithms.
% 
% This example shows a streaming fixed-point implementation of the fog 
% rectification algorithm that is suitable for deployment to hardware.
%
% To improve the foggy input image, the algorithm performs fog removal and
% then contrast enhancement. The diagram shows the steps of both these
% operations. 
%
% This example takes a foggy RGB image as input. To perform fog removal, 
% the algorithm estimates the dark channel of the image, calculates the 
% airlight map based on the dark channel, and refines the airlight map by 
% using filters. The restoration stage creates a defogged image by 
% subtracting the refined airlight map from the input image.
% 
% Then, the Contrast Enhancement stage assesses the range of intensity 
% values in the image and uses contrast stretching to expand the range 
% of values and make features stand out more clearly. 
%
% <<../FogRectificationExampleBlockDiagram.PNG>>
%
%% Fog Removal
%
% There are four steps in performing fog removal.
% 
% 1. *Dark Channel Estimation*: The pixels that represent the non-sky region
% of an image have low intensities in at least one color component. The 
% channel formed by these low intensities is called the _dark channel_.
% In a normalized, fog-free image, the intensity of dark channel pixels is
% very low, nearly zero. In a foggy image, the intensity of dark channel 
% pixels is high, because they are corrupted by fog. So, the fog removal 
% algorithm uses the dark channel pixel intensities to estimate the amount 
% of fog.
%
% The algorithm estimates the dark channel $$ I^c_{dark}(x,y) $$ by finding
% the pixel-wise minimum across all three components of the input image 
% $$ I^c_{}(x,y) $$ where $$ c\ \epsilon\ [r,g,b] $$.
%
% 2. *Airlight Map Calculation*: The whiteness effect in an image is known
% as _airlight_. The algorithm calculates the airlight map from the dark
% channel estimate by multiplying by a haze factor, $z$, that 
% represents the amount of haze to be removed. The value of $z$ is between
% 0 and 1. A higher value means more haze will be removed from the image.
%
% $$ I_{air}(x,y) = z \times \min_{c\ \epsilon\ [r,g,b]} I^c_{dark}(x,y)$$
% 
% 3. *Airlight Map Refinement*: The algorithm smoothes the airlight image 
% from the previous stage by using a Bilateral Filter block. This smoothing 
% strengthens the details of the image. The refined image is referred to as 
% $I_{refined}(x,y)$.
%
% 4. *Restoration*: To reduce over-smoothing effects, this stage corrects the 
% filtered image using these equations. The constant, $m$, represents 
% the mid-line of changing the dark regions of the airlight map from 
% dark to bright values. The example uses an empirically derived value 
% of $m=0.6$.
% 
% $$ I_{reduced}(x,y)  =  m \times \ min ({I_{air}}(x,y) , I_{refined}(x,y)) $$
%
% The algorithm then subtracts the airlight map from the input foggy image and
% multiplies by the factor $$  \frac{255}{255-I_{reduced}(x,y)} $$. 
%
% $$ I_{restore}(x,y) = 255 \times \frac{I^c_{}(x,y)-I_{reduced}(x,y)}{255-I_{reduced}(x,y)} $$
%
%% Contrast Enhancement
%
% There are five steps in contrast enhancement.
%
% 1. *RGB to Gray Conversion*: This stage converts the defogged RGB image,
% $$ I^c_{restore}(x,y) $$, from the fog removal algorithm into a grayscale 
% image, $$ I_{gray}(x,y) $$.
%
% 2. *Histogram Calculation*: This stage uses the Histogram block to count 
% the number of pixels falling in each intensity level from 0 to 255. 
%
% 3. *Histogram Normalization*: The algorithm normalizes the histogram values 
% by dividing them by the input image size.
%
% 4. *CDF Calculation*: This stage computes the cumulative distribution 
% function (CDF) of the normalized histogram bin values by adding them to
% the sum of the previous histogram bin values.
%
% 5. *Contrast Stretching*: Contrast stretching is an image enhancement 
% technique that improves the contrast of an image by stretching the
% range of intensity values to fill the entire dynamic range. When dynamic
% range is increased, details in the image are more clearly visible.
%
% 5a. _i1 and i2 calculation_: This step compares the CDF values with two
% threshold levels. In this example, the thresholds are 0.05 and 0.95. This
% calculation determines which pixel intensity values align with the CDF
% thresholds. These values determine the intensity range for the stretching
% operation. 
% 
% 5b. _T calculation_: This step calculates the stretched pixel intensity values
% to meet the desired output intensity values, $$o_1 $$ and $$ o_2 $$. 
%
% $$ o_1 $$ is 10% of maximum output intensity floor(10*255/100) for |uint8| input. 
%
% $$ o_2 $$ is 90% of maximum output intensity floor(90*255/100) for |uint8| input.
%
% _T_ is a 256-element vector divided into segments $$ t_1 $$, $$ t_2 $$, 
% and $$ t_3 $$. The segment elements are computed from the relationship 
% between the input intensity range and the desired output intensity range. 
%
% <<../FRlinearTransform.png>>
%
% $$ i_1 $$ and $$ i_2 $$ represent two pixel intensities in the input
% image's range and $$ o_1 $$ and $$ o_2 $$ represent two pixel intensities
% in the rectified output image's range.
%
% These equations show the how the elements in _T_ are calculated.
%
% $$ t_1 = \frac{o_1}{i_1}[0:i_1]  $$
%
% $$ t_2 = (((\frac{o_2-o_1}{i_2-i_1})[(i_1+1):i_2])  -
% ((\frac{o_2-o_1}{i_2-i_1})i_1)) + o_1 $$
%
% $$ t_3 = (((\frac{255-o_2}{255-i_2})[(i_2+1):255])  -
% ((\frac{255-o_2}{255-i_2})i_2)) + o_2  $$
%
% $$ T = [t_{1} \quad  t_{2} \quad  t_{3}] $$
%
% 5c. _Replace intensity values_: This step converts the pixel intensities
% of the defogged image to the stretched intensity values. Each pixel 
% value in the defogged image is replaced with the corresponding intensity in _T_.
%

%% HDL Implementation
%
% The example model implements the algorithm using a 
% streaming pixel format and fixed-point blocks from Vision HDL Toolbox(TM). 
% The serial interface mimics a real time system and is efficient 
% for hardware designs because less memory is required to store pixel data 
% for computation. The serial interface also allows the design
% to operate independently of image size and format and makes it more 
% resilient to timing errors. Fixed-point data types use fewer
% resources and give better performance on FPGA. The necessary variables 
% for the example are initialized in the *InitFcn* callback. 
%
open_system('FogRectificationHDL');
set(allchild(0),'Visible','off');
%% 
% The FogImage block imports the input image to the model. The <docid:visionhdl_ref#bt4qnrk-1 Frame To 
% Pixels> block converts the input frames to a 
% pixel stream of |uint8| values and a |pixelcontrol| bus. The <docid:visionhdl_ref#bt4qnzd-1 Pixel To 
% Frame> block converts the pixel stream back to image frames. The 
% hdlInputViewer subsystem and hdlOutputViewer subsystem show 
% the foggy input image and the defogged enhanced output image, respectively.
% The ImageBuffer subsystem stores the defogged image so the 
% Contrast Enhancement stages can read it as needed.
% 
% The FogRectification subsystem includes the fog removal and contrast 
% enhancement algorithms, implemented with fixed-point datatypes. 
%
open_system('FogRectificationHDL/FogRectification/FogRemoval');
%%
% In the FogRemoval subsystem, a Minimum block named DarkChannel
% calculates the dark channel intensity by finding the minimum across
% all three components. Then a <docid:visionhdl_ref#mw_3631b6e8-b6bb-469a-b01e-e090863ccb7d Bilateral
% Filter> block refines the dark channel results. The filter block has
% the spatial standard deviation set to |2| and the intensity standard deviation
% set to |0.5|. These parameters are used to derive the filter coefficients.
% The bit width of the output from filter stage is the same as that
% of the input. 
%
% Next, the airlight image is calculated by multiplying the refined 
% dark channel with a haze factor, |0.9|. Multiplying by this factor after
% the bilateral filter avoids precision loss that would occur from truncating
% to the maximum 16-bit input size of the bilateral filter. 
% 
% The Restoration subsystem removes the airlight from the image and then 
% scales the image to prevent over-smoothing. The <docid:visionhdl_ref#bvkomp_ Pixel Stream Aligner> 
% block aligns the input pixel stream with the airlight image before subtraction. 
% The scale factor, $$ m $$, is found from the midpoint of the difference
% between the original image and the image with airlight removed. 
% The Restoration subsystem returns a defogged image that has low contrast. 
% So, contrast enhancement must be performed on this image to increase the
% visibility.
%
% The output from the FogRemoval subsystem is stored in the Image Buffer. The 
% ContrastEnhancement subsystem asserts a |pop| signal to read
% the frame back from the buffer. 
%
open_system('FogRectificationHDL/FogRectification/ContrastEnhancement');
%%
% The ContrastEnhancement subsystem uses the 
% <docid:visionhdl_ref#bucjt42-1 Color Space converter> block to convert
% the RGB defogged image to a grayscale image. Then the Histogram block 
% computes the histogram of pixel intensity values. When the histogram is 
% complete, the block generates a *readRdy* signal. Then the HistNormAndCDFCalculation
% subsystem normalizes the histogram values and computes the CDF.
%
% The i1Andi2Calculation subsystem computes the $$ i_1 $$ and $$ i_2 $$ 
% values that describe the input intensity range. Then the TCalculation 
% subsystem returns the list of target output intensity values. These 256 values 
% are written into a lookup table. The logic in the Contrast Stretching-LUT
% area generates a *pop* signal to read the pixel intensities of the 
% defogged image from the Image Buffer, and feeds these values as read 
% addresses to the LUT. The LUT returns the corresponding stretched 
% intensity values defined in _T_ to replace the pixel values in the defogged image.  
%
open_system('FogRectificationHDL/ImageBuffer');
%%
% The Image Buffer subsystem contains two options for modeling the connection 
% to external memory. It is a variant subsystem where you can select between 
% the BehavioralMemory subsystem and the Video Frame Buffer block. 
%
% <<../FRvariantSubsystem.PNG>>
%
% Use the BehavioralMemory subsystem if you do not have the support package mentioned
% below. This block contains HDL FIFO blocks. The BehavioralMemory returns the 
% stored frame when it receives a pop request signal. The pop request to 
% BehavioralMemory must be high for every row of the frame.
%
% The Video Frame Buffer block requires the SoC Blockset(TM) Support
% Package for AMD(R) SoC Devices. With the proper reference 
% design, the support package can map this block to an AXI-Stream VDMA buffer
% on the board. This frame buffer returns the stored frame when it receives
% the popVB request signal. The pop request to this block must be high only
% one cycle per frame.
%
% The inputs to the Image Buffer subsystem are the pixel stream and control
% bus generated after fog removal. The pixel stream is fetched during the
% Contrast Enhancement operation, after the stretched intensities (_T_) are
% calculated.
%
%% Simulation and Results 
%
% This example uses an RGB 240-by-320 pixel input image.
% Both the input pixels and the enhanced output pixels use the |uint8|
% data type. This design does not have multipixel support.
% 
% The figure shows the input and the enhanced output 
% images obtained from the FogRectification subsystem.
%
% <<../FogRectificationResult.png>>
%
% You can generate HDL code for the FogRectification subsystem. An HDL Coder(TM)
% license is required to generate HDL code. This design was synthesized for
% the Intel(R) Arria(R) 10 GX (115S2F45I1SG) FPGA. The table
% shows the resource utilization. The HDL design achieves a clock rate
% of over 200 MHz.
%
%   % ===============================================================
%   % |Model Name              ||      FogRectificationHDL     ||
%   % ===============================================================
%   % |Input Image Resolution  ||         320 x 240            ||
%   % |ALM Utilization         ||           10994              ||
%   % |Total Registers         ||           20632              ||
%   % |Total RAM Blocks        ||            67                ||
%   % |Total DSP Blocks        ||            39                ||       
%   % ===============================================================
%
% Copyright 2019 The MathWorks, Inc.


