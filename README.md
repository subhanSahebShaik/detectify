# Abstract

The growing computation power has made the deep learning algorithms so powerful that creating an indistinguishable human synthesized video popularly called as deep fakes have become very simple. Scenarios where this realistic face swapped deep fakes are used to create political distress, fake terrorism events, blackmail peoples are easily envisioned. In this work, we describe a new deep learning-based method that can effectively distinguish AI-generated fake videos from real videos. Our method is capable of automatically detecting the replacement and reenactment deep fakes. Our system uses a Res-Next Convolution neural network to extract the frame- level features and these features and further used to train the Long-Short-Term Memory (LSTM) based Recurrent Neural Network (RNN) to classify whether the video is subject to any kind of manipulation or not, i.e. whether the video is deep fake or real video. To emulate the real time scenarios and make the model perform better on real time data, we evaluate our method on large amount of balanced and mixed data-set prepared by mixing the various available data-set like Face-Forensic++, Deepfake detection challenge, and Celeb-DF.

# How to Run

1. Open Terminal or Command prompt from the location where you want to set the project.
2. Now run the following command,
> git clone https://github.com/subhanSahebShaik/detectify.git

It only works if Git installed in your system, else download zip file manually from GitHub and extract it into desired location.

3. Now run,
> pip install -r requirements.txt

4. Now run server,
> python manage.py runserver

5. Now local server is ready. Open any browser and visit http://127.0.0.1:8000/

# Special Thanks To

I would like to express my special thanks to [abhijitjadhav1998/Deepfake_detection_using_deep_learning](https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning) repository for providing valuable insights and inspiration for this project. The code and ideas presented in this repository served as a foundation and guide for developing our deep learning-based method to detect AI-generated fake videos.

---

[![Original Repository](https://img.shields.io/badge/Original%20Repository-abhijitjadhav1998/Deepfake_detection_using_deep_learning-blue?logo=github)](https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning)
