from time import time

#from here: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
#ASSUMES BOXES are in (x1,y1,x2,y2) format
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
    
def toCornerForm(box):
	x,y,w,h = box
	return (x, y, x+w, y+h)

    
def trackIOUSimple(detections, numFrames, frameWidth, frameHeight):
    #TODO: Implement	
    # track = (frame_idx, box)
    # Initialize the allTracks list with empty lists for each frame
    # allTracks = [[frame_idx, box), ...], [...], ...]
    allTracks = [] # List of lists
    trackBoxes_indices = None # List of boxes with confidence level of 0.7 and are added
    CONFIDENCE_THRESHOLD = 0.7
    MINIMUM_THRESHOLD = 0.1
    
    for frameIdx in range(len(detections)):	
        # All the tracks on the current frameIdx
        # TODO: Fix the indexing of all the Tracks
        for trackIdx in range(len(allTracks[frameIdx])):
        # detections = [[((x, y, w, h), score)], [...], ...]
            for detectonIdx in range(len(detections[trackIdx])):
                # Create dict for curr maxBox
                max_details = {
                    'maxConfidence': 0.0,
                    'maxBox': None,
                    'maxFrame': None, 
                }
                
                # Create dict for previous maxBox
                max_previous_details = { 
                    'prevConfidence': 0.0,
                    'prevBox': None,
                    'prevFrame': None, 
                }
                
                iou = None
                lastBox = None
                currBox = None
                currConfidence = None
                
                defaultBox = ((0, 0, 0, 0), 0)
                trackBoxes_indices = [] # For checking if the box is already in the trackBoxes, with confidence level of 0.7 and are added
                
                # check if the track is empty
                if len(allTracks[trackIdx]) == 0:
                    # TODO: LIKE HOW WE GONNA ADD THE TRACK?
                    # Initialize the track with the first detection
                    track = []
                else: 
                    lastBox = allTracks[trackIdx][-1][1] # Last box in the track
                    currBox = detections[frameIdx][detectonIdx][0] # Current box in the detection
                    currConfidence = detections[frameIdx][detectonIdx][1] # Current confidence level
                    
                    # Compute the IOU
                    iou = bb_intersection_over_union(lastBox, currBox)
                         
                frame = frameIdx + 1 
                
                # Check if the score is greater than the maxConfidence
                if max_details['maxBox'] is None or (iou > max_details['maxIOU'] and iou > 0.1):
                    # Obtain the previous maxBox
                    max_previous_details['prevConfidence'] = max_details['maxConfidence']
                    max_previous_details['prevBox'] = max_details['maxBox']
                    max_previous_details['prevFrame'] = max_details['maxFrame']
                    
                    # If the current box has a higher confidence than the previous maxBox, update the maxBox
                    max_details['maxConfidence'] = currConfidence
                    # Update the maxBox of the track with the current box
                    max_details['maxBox'] = currBox 
                    max_details['maxFrame'] = frame
                    max_details['maxIOU'] = iou 
                    
                    allTracks[trackIdx].append((frame, max_details['maxBox']))
                    
                    # Add the box to the trackBoxes, which will not be used since it is already replaced by maxBox
                    if (max_previous_details['maxConfidence'] >= CONFIDENCE_THRESHOLD):
                        trackBoxes_indices.append((frame, max_previous_details['prevBox']))
                else: 
                    # If there are no detected boxes with more than 0.1 IOU with the last box, stop the track 
                    allTracks[trackIdx].append((frame, defaultBox))    
                    
        # If the trackBoxes is not empty, add the current box to the track
        if len(trackBoxes_indices) > 0:
            allTracks[frameIdx].append(trackBoxes_indices)
        
    return allTracks

   
def track_medflow_matlab_wrapper(detections, numFrames, frameWidth, frameHeight,
                                 imgPath):

    start = time()
    allTracks = trackIOUSimple(detections, numFrames, frameWidth, frameHeight)
    end = time()

    idx = 1
    out = []
    for track in allTracks:
        for (frame, box) in track:
            (x1, y1, w,h) = box
            out.append([float(x1), float(y1), float(w), float(h),
                        float(frame), float(idx)])
        idx += 1

    num_frames = len(detections)

    # this part occasionally throws ZeroDivisionError when
    # evaluated in our code
    try:
        speed = num_frames / (end - start)
    except ZeroDivisionError:
        speed = num_frames / 0.1

    return (speed, out)
