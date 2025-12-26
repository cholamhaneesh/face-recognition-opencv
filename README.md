# Face Recognition using OpenCV and KNN

This project performs real-time face recognition using webcam input.
Face data is collected and stored as `.npy` files, and recognition is done
using a manually implemented KNN algorithm.

## Features
- Face detection using Haar Cascades
- Face data collection and storage
- Manual KNN implementation (Euclidean distance)
- Real-time face recognition with name display

## Tech Stack
- Python
- OpenCV
- NumPy

## How it Works
1. Run `Face_Data_Collection.py` to collect face data.
2. Face data is saved as `.npy` files using the person's name.
3. Run `Face_Recognition.py` to recognize faces in real time.

## Algorithm
- KNN (manually implemented)
- Distance: Euclidean
- Voting-based classification
