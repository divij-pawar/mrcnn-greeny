#######################################################################
import numpy as np
#Apply mask to image
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

#Add two masks together
def add_mask(current_mask, mask=None):
  if mask is None:
    print("Returned")
    return current_mask

  assert current_mask.shape == mask.shape
  mask = np.logical_or(mask,current_mask)

  print(mask.shape)
  return mask
  
#######################################################################

# This function is used to show the object detection result in original image.
def display_instances(image_c, boxes, masks, ids, names, scores):

    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        # Check if number of boxes,masks,ids is same
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    mask = None

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        # use label to select person object from all the 80 classes in COCO dataset
        label = names[ids[i]]
        if label == 'person':
            # save the largest object in the image as main character
            # other people will be regarded as background
            current_mask = masks[:, :, i]

            mask = add_mask(current_mask, mask)
        else:
            continue

        # apply mask for the image
    # by mistake you put apply_mask inside for loop or you can write continue in if also

    image_c = apply_mask(image_c, mask, (0.0,1.0,0.0))
    return image_c
    
    
