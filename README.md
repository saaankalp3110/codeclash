# codeclash

Group Members: Jacob Lister, Wyatt Shepherd, Dominik Shramek, Sankalp Sinha

We worked on the Software Part (AI-Based Processing & Detection)

1 -> Machine Learning-Based Object Detection in Foggy and Rainy Conditions   
2 -> AI-Powered Risk Assessment and Adaptive Decision-Making System   

1) We made use of the Foggy Cityscapes dataset mentioned in the problem statement file and also generated synthetic data for smaller test cases.
   We made use of YOLO to test some pretrained models which help in derain, defog and denoise.
   We made use of MATLAB to implement the traditional methods.
   We have a comparison file which makes use of metrics like mAP and IoU as mentioned in the problem statement file.

2) We made use of YOLO to come up with a model which does object detection in real-time. We have not put the option of the label but the code
   is present which can implment that. The code tracks the movement trajectory and can also find the relative speed of the vehicles. It uses a threshold value
   which indicate the collision risk and shows it on the top left and colors the bounding box as a visual indiction of the object that you need to be careful about.
   We were not able to find a dataset that would suit all the requirements.
   NOTE:
   The code is from first-person POV.
   
   

References:

@inproceedings{Zamir2021MPRNet,
    title={Multi-Stage Progressive Image Restoration},
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat
            and Fahad Shahbaz Khan and Ming-Hsuan Yang and Ling Shao},
    booktitle={CVPR},
    year={2021}
}


