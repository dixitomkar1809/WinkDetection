# Wink Detection
**Author: Omkar Dixit**
Basic approach here is to detect face first, when the face detection is done, we then detect eyes in the face region (for obvious reasons)

Since the shape of eyes, specifically the outline is similar to that of mouth, so rather than taking the face as our region of interest, we take the upper part of the face as our region of interest, thus ignoring the mouth to be detected as eye !

Now, that we have the proper region to detect eye, we now count the number of eyes, after that we count the number of detection
if its 1 then its a wink if not then not a wink.

Now for the cascades I have used the alt face detection and eyes glasses cascade, i found that working better for the lot of images i had.



