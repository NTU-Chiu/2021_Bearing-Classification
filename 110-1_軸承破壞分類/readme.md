# What the project does?
Using machine learning to classify different bearing damage situation.
## Input
* Vibration signal:
  ![image](https://github.com/NTU-Chiu/ML_Projects/assets/91785016/21accb61-b85e-4794-93a1-9e3a4f07e992)
* Rotation signal:
  ![image](https://github.com/NTU-Chiu/ML_Projects/assets/91785016/6fb34b07-77d0-467f-8ae2-8e70aea8549f)

## Output
* Bearing damage situation:
* Class 0: Healthy bearing
* Class 1: Outer bearing damage
* Class 2: Inner bearing damage
## Results
* 100 % accuracy
## Technique
* Preprocessing:
  Using Fast Fourier Transform to transform the time domain signals to frequency domain signals.
  ![image](https://github.com/NTU-Chiu/ML_Projects/assets/91785016/9da3af00-4f42-430c-89b5-2290ae2cd605)
  ![image](https://github.com/NTU-Chiu/ML_Projects/assets/91785016/cf377d8c-832c-4e4b-be39-7ba0f876b919)

* Model:
  Using 1D CNN to extract fearures of signals.
  ![image](https://github.com/NTU-Chiu/ML_Projects/assets/91785016/a4da5184-2b65-489c-a277-58e3632a534b)

# Reference:
* This project is the final project of diginal signal processing class at NTU.
* The dataset is private.
* This code is referenced from the TA, Bo Han Kung.

