# FaceTracking

[forked from https://github.com/marco7877/LipReading]

This scripts generates x,y coordinates tracking 68 face landmarks.

Face (Viola and Jones, 2001), and 68 facial landmarks (Kazemi and Sullivan, 2014)
are detected at the same time as the aforementioned. 68 facial landmarks are tracked and stored for each frame. 20 landmarks. 
Output: 
- Video with face tracking for visual inspection
- CSV file with x,y coordinates for 68 facial landmarks for each frame (when face is not detected the coordinates are NaNs)
- CSV file with frame numbers
