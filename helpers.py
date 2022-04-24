#######################################################################
import numpy as np
#Apply mask to image frame. 
# i.e paint green everything except people
# image_c - Received image to be painted
# mask - Received mask of (multiple) people
# color - color choice for background
# alpha - transparency variable: set to 1 for no transparency
def apply_mask(image_c, mask, color, alpha=1):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image_c[:, :, c] = np.where(mask == 1,
                                  image_c[:, :, c],
                                  image_c[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255)
    return image_c
    
#######################################################################

#Add two masks together. Used to add together different 
# instances of people together to stitch into one mask
# current_mask - mask for the current instance of object (person)
# mask - mask of all previous people combined
def add_mask(current_mask, mask=None):
  if mask is None:
    print("Returned")
    return current_mask

  assert current_mask.shape == mask.shape
  mask = np.logical_or(mask,current_mask)

  print(mask.shape)
  return mask
  
#######################################################################

# This function is used to overlay the object detection result in original image.
# image_c - image_c - Received image to be processed
# boxes - the coordinates of boxes of each object detected
# masks - the masks of each individual object (instances) detected
# ids - the unique ids of each instances detected in image
# names - all the classes which can be detected by model
# scores - individual scores of each instances
def display_instances(image_c, boxes, masks, ids, names, scores):

    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]
    
    #Error check if no instances 
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        # Check if number of boxes,masks,ids is same
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    mask = None
    
    # Iterate over all instances detected to find all people
    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        # use label to select person object from all the 80 classes in COCO dataset
        label = names[ids[i]]
        if label == 'person':
            # Select complete mask for current instance
            current_mask = masks[:, :, i]
            # Add current mask to final mask
            mask = add_mask(current_mask, mask)
        else:
            continue

    # define BGR value of green for green bg 
    green_color = (0.0,1.0,0.0)
    # apply final mask for the image
    image_c = apply_mask(image_c, mask, green_color)
    return image_c
    
    
