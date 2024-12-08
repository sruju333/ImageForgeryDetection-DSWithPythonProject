# Image Forgery Detection  

**Author**: Srujana Niranjankumar  
**Course**: Data Science with Python  
**Institution**: Boston University Metropolitan College  
**BU ID**: U61717332  

---

## Project Description  

This project implements an **Image Forgery Detection** system to identify manipulated regions within an image using deep learning techniques. The goal is to create a robust pipeline for detecting various types of image forgeries, such as copy-move, splicing, or retouching, by leveraging convolutional neural networks (CNNs) and preprocessing algorithms.  

---

## Features  

1. **Dataset Collection**:  
   - Collect and curate a dataset representing real-world image manipulation scenarios.  

2. **Data Preprocessing**:  
   - Clean and enhance image quality using preprocessing algorithms.  

3. **Model Training**:  
   - Train a CNN model for forgery detection using the preprocessed dataset.  

4. **Model Saving**:  
   - Save the trained model for reuse and deployment.  

5. **User Interaction**:  
   - Allow users to upload an image for analysis.  

6. **Forgery Prediction**:  
   - Predict whether the uploaded image contains manipulated content.  

---

## Technologies Used  

- **Programming Language**: Python  
- **Libraries/Frameworks**:  
  - TensorFlow: For building and training the CNN model.  
  - OpenCV: For image preprocessing and manipulation detection.  
  - NumPy, Pandas: For data manipulation and analysis.  
  - Flask: To deploy the web application.

---


### Installation

pip install tensorflow opencv-python flask tflearn numpy tqdm

Install Pillow by following instructions from [here](https://pillow.readthedocs.io/en/latest/installation/building-from-source.html#building-from-source)
Install the lib packages required for Pillow first. 
Then install Pillow 9.5.0:

```bash
pip install Pillow==9.5.0
```

#### Run on Terminal

Run below commands in terminal to open a virtual python env and run the app - 

$ python3 -m venv myenv
$ source myenv/bin/activate
$ python3.9 app.py