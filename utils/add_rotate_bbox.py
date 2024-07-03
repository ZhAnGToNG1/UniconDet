import numpy as np
import cv2

color_category = np.array([[0,0,255],[0,255,0],[2,124,50],[0,128,0],[255,0,0],[128,0,255],[0,255,0],[255,128,0],[128,255,0],
                           [255,0,255],[255,0,128],[0,255,0],[2,202,12],[0,255,255],[0,255,128],[145,240,2],[245,90,150],[211,4,4],
                           [222,111,22],[234,12,33]])
# categories = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter','container-crane']

categories = ["airplane","ship","storage-tank","baseball-diamond",
                           "tennis-court","basketball-court","ground-track-field","harbor","bridge","vehicle"]

isaid_classes = ['ship',
            'storage_tank',
            'baseball_diamond',
            'tennis_court',
            'basketball_court',
            'Ground_Track_Field',
            'Bridge',
            'Large_Vehicle',
            'Small_Vehicle',
            'Helicopter',
            'Swimming_pool',
            'Roundabout',
            'Soccer_ball_field',
            'plane',
            'Harbor']

def draw_bbox(bbox,image,cat,show_Txt = True):
    bbox = np.array(bbox,dtype=np.float32)

    x1,y1,x2,y2,x3,y3,x4,y4,score = bbox

    cat = int(cat)
    label = categories[cat]
    score = str(round(float(score),2))

    c = color_category[cat]
    c = c.tolist()

    cv2.line(image, (x1,y1),  (x2,y2),c,1)
    cv2.line(image, (x2, y2), (x3, y3), c, 1)
    cv2.line(image, (x3, y3), (x4, y4), c, 1)
    cv2.line(image, (x4, y4), (x1, y1), c, 1)
    cv2.putText(image, score + '_' + label, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)


    return image


def draw_bbox_for_seg(contour, image, cat, show_Txt = True):
    contour = np.array(contour,dtype=np.float32)
    score = contour[-1]
    poly = contour[:360].reshape(-1,2)
    score = str(round(float(score),2))
    x1, y1 = poly[0]
    cat = int(cat)
    label = isaid_classes[cat]
    cv2.putText(image, score + '_' + label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1,
                lineType=cv2.LINE_AA)

    # cat = int(cat)
    # label = categories[cat]
    # score = str(round(float(score),2))
    #
    # c = color_category[cat]
    # c = c.tolist()

    for point in poly:
        point = point.astype('int16')
        cv2.circle(image, tuple(point), 1, (0, 255, 0), 1)
    return image
