# What the project does?
Using machine learning to classify different bearing damage situation.
## Input
* Vibration and ratation signals: <br />
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/3233511b-c8d7-4de5-9fd1-b64cf437f3e4.png" width = "250" height = "250">

## Output
* Bearing damage situation:  <br />
<img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/b50ab5ea-1e15-41ed-9037-cb3258267bda.png"  width = "750" height = "250">

## Results
* 100 % accuracy
## Technique
* Preprocessing:
  Using Fast Fourier Transform to transform the time domain signals to frequency domain signals. <br />
  <img src = "https://github.com/NTU-Chiu/ML_Projects/assets/91785016/80595607-f10b-4223-982d-ab32cbb9edb" width = "250" height = "250">

* Model:
  Using 1D CNN to extract fearures of signals.  <br />
![image](https://github.com/NTU-Chiu/ML_Projects/assets/91785016/a4da5184-2b65-489c-a277-58e3632a534b)

# Reference:
* This project is the final project of diginal signal processing class at NTU.
* The dataset is private.
* This code is referenced from the TA, Bo Han Kung.

