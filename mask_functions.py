import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def square_ROI(image, message_num=6):
    """
    This function will create a square mask based on the corners the user selects
    :param image: 2D ndarray
                The image to mask
    :return: The image mask
    """
    # Open the image and click the 4 corners
    coords = click_image(image, message_num=message_num)

    # Array to hold the saved mask
    num_rows, num_cols = np.shape(image)

    # Create the mask
    mask = rectangular_mask(coords, [num_rows, num_cols])

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    coords = np.squeeze(coords)
    x_max = int(round(np.max(coords[:, 1])))
    x_min = int(round(np.min(coords[:, 1])))

    y_max = int(round(np.max(coords[:, 0])))
    y_min = int(round(np.min(coords[:, 0])))

    corner = (y_min-0.5, x_min-0.5)
    height = y_max - y_min + 1
    width = x_max - x_min + 1
    sq = Rectangle(corner, height, width, fill=False, edgecolor='red')
    ax.add_artist(sq)

    plt.show()
    plt.pause(2)
    plt.close()

    return mask


def phantom_ROIs(image, radius=6, message_num=3):
    """
    This function generates the number of circular ROIs corresponding to user input center points of each ROI
    It will output as many ROIs as coordinates clicked
    The radius is also set by the user and may need some fine tuning
    Each mask will have 1's inside the ROI and nan everywhere else
    :param image: The image as a numpy array
    :param radius: The desired ROI radius (all ROIs will have this radius)
    :return: the saved masks as a single numpy array (individual masks callable by masks[i]
    """

    # Open the image and click the 6 ROIs
    coords = click_image(image, message_num=message_num)

    # Array to hold the saved masks
    num_rows, num_cols = np.shape(image)
    num_of_ROIs = len(coords)
    masks = np.empty([num_of_ROIs, num_rows, num_cols])

    # Plot to verify the ROI's
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    for idx, center in enumerate(coords):

        # Make the mask for those center coordinates
        masks[idx] = circular_mask(center, radius, (num_rows, num_cols))

        # Verify the ROI
        circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
        ax.add_artist(circ)

    plt.show()
    plt.pause(2)
    plt.close()

    return masks


def single_circular_ROI(image):
    """
    This function takes a single ROI representing the background of an image, you will click the center of the ROI and
    a point along its radius
    :param image: The image as a numpy array
    :return: mask of the ROI containing 1's inside the ROI and nan elsewhere
    """

    # Open the image and click the single background ROI
    coords = click_image(image, message_num=1)

    # Array to hold the saved mask
    num_rows, num_cols = np.shape(image)

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    # Get the center point and the point on the edge of the desired ROI
    center = coords[0]
    point = coords[1]
    x1 = center[0]
    y1 = center[1]
    x2 = point[0]
    y2 = point[1]
    radius = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    mask = circular_mask(center, radius, (num_rows, num_cols))

    # Verify the ROI
    circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
    ax.add_artist(circ)

    plt.show()
    plt.pause(2)
    plt.close()

    return mask


def entire_phantom(image, radii=13):
    """
    This function will take an initial mask of the entire phantom based on the user selecting the center and the outer
    edge, in that order. Each of the vial centers will then be selected to subtract them from the greater circle.
    The final mask will only encompass the phantom body
    :param image: The image to draw on
    :param radii: The radii of the inner
    :return:
    """
    ## OUTER PHANTOM
    coords1 = click_image(image, message_num=5)

    # Array to hold the saved mask
    num_rows, num_cols = np.shape(image)

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    # Get the center point and the point on the edge of the desired ROI
    center = coords1[0]
    point = coords1[1]
    x1 = center[0]
    y1 = center[1]
    x2 = point[0]
    y2 = point[1]
    radius = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    outer_mask = circular_mask(center, radius, (num_rows, num_cols))
    circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
    ax.add_artist(circ)

    ## VIALS
    coords2 = click_image(image, message_num=2)
    num_of_ROIs = len(coords2)
    masks = np.empty([num_of_ROIs, num_rows, num_cols])

    for idx, center in enumerate(coords2):
        # Make the mask for those center coordinates
        masks[idx] = circular_mask(center, radii, (num_rows, num_cols))

        # Verify the ROI
        circ = plt.Circle(center, radius=radii, fill=False, edgecolor='red')
        ax.add_artist(circ)

    plt.show()
    plt.pause(3)
    plt.close()
    # Create the full mask
    for mask in masks:
        inner = np.zeros([num_cols, num_rows])
        inner[mask != 1.0] = 1
        inner[mask == 1.0] = np.nan
        outer_mask = np.multiply(outer_mask, inner)

    return outer_mask


def air_mask(image):
    """
    This function takes an image of the circular phantom and creates a mask for only the air outside of the phantom
    body
    :param image: The image to draw on
    :return: The air mask
    """

    coords1 = click_image(image, message_num=5)

    # Size of the image
    num_rows, num_cols = np.shape(image)

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    # Get the center point and the point on the edge of the desired ROI
    center = coords1[0]
    point = coords1[1]
    x1 = center[0]
    y1 = center[1]
    x2 = point[0]
    y2 = point[1]
    radius = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    mask = circular_mask(center, radius, (num_rows, num_cols))
    circ = plt.Circle(center, radius=radius, fill=False, edgecolor='red')
    ax.add_artist(circ)

    plt.show()
    plt.pause(1)
    plt.close()

    # Switch the mask to return the air outside of the phantom
    switch = np.zeros([num_cols, num_rows])
    switch[mask != 1.0] = 1
    switch[mask == 1.0] = np.nan

    return switch


def single_pixels_mask(image):
    """
    This function will create a mask just to obtain individual pixel values, the pixels that are clicked
    :param image: The image to mask
    :return: The mask of the clicked pixels
    """

    coords = click_image(image, message_num=7)  # The coordinates of the pixels to mask

    # Size of the image
    num_rows, num_cols = np.shape(image)

    # Plot to verify the ROI's
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')

    mask = np.ones([num_rows, num_cols])*np.nan  # Create a matrix with only nan values

    # Get each pair of coordinates
    for pair in coords:
        x = int(round(pair[1]))
        y = int(round(pair[0]))
        mask[x, y] = 1  # Set each clicked pixel equal to one

        # Verify that the correct pixel has been chosen
        corner = (y - 0.5, x - 0.5)
        height = 1
        width = 1
        sq = Rectangle(corner, height, width, fill=False, edgecolor='red')
        ax.add_artist(sq)

    plt.show()
    plt.pause(2)
    plt.close()

    return mask


def click_image(image, message_num=12, message=None):
    """

    :param image:
    :param message_num:
    :return:
    """
    # These are the possible instructions that will be set as the title of how to collect the desired points
    instructions = {0: 'Click the center of the phantom first, then a point giving the desired radius from the center'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    1: 'Click the center of the desired ROI, then the desired radius (relative to the center)'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    2: 'Click the center of each ROI in order from water to highest concentration.'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    3: 'Click the center of each ROI from water vial, then move counter-clockwise.'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    4: 'Click the centers of the desired ROIs'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    5: 'Click the center of the phantom and the edge of the phantom'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    6: 'Click four corner pixels that form a rectangle/square.'
                       '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    7: 'Click the desired pixels to obtain values from.'
                       '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    8: 'AIR ROI: Click four corner pixels that form a rectangle/square.'
                       '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    9: 'WATER ROIs: Click the center of each water vial.'
                       '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    10: 'PHANTOM ROIs: Click the a bunch around the phantom body.'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    11: 'CONTRAST ROIs: click the contrast vials, starting from highest to lowest concentration.'
                        '\n Left-click: add point, Right-click: remove point, Enter: stop collecting',
                    12: 'Click the desired points'
                    }

    pix = int(len(image)/3)
    # med = np.nanmedian(image[pix:-pix, pix:-pix])
    med = np.nanmedian(image)
    vmin = med - (7*np.abs(med))
    vmax = med + (7*np.abs(med))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(image, vmin=vmin, vmax=vmax)
    # ax.imshow(image, vmin=-500, vmax=3000)
    if message_num:
        ax.set_title(instructions[message_num])
    elif message:
        ax.set_title(message)

    # Array to hold the coordinates of the center of the ROI and its radius
    # Left-click to add point, right-click to remove point, press enter to stop collecting
    coords = plt.ginput(n=-1, timeout=-1, show_clicks=True)
    coords = np.array(coords)
    coords = np.round(coords, decimals=0)
    plt.close()

    return coords


def circular_mask(center, radius, img_dim):
    """
    Creates a mask matrix of a circle at the specified location and with the specified radius
    :param center:
    :param radius:
    :param img_dim:
    :return:
    """
    # Create meshgrid of values from 0 to img_dim in both dimension
    xx, yy, = np.mgrid[:img_dim[0], :img_dim[1]]

    # Define the equation of the circle that we would like to create
    circle = (xx - center[1])**2 + (yy - center[0])**2

    # Create the mask of the circle
    arr = np.ones(img_dim)
    mask = np.ma.masked_where(circle < radius**2, arr)
    mask = mask.mask

    arr = np.zeros([img_dim[0], img_dim[1]])
    arr[mask] = 1
    arr[arr == 0] = np.nan

    return arr


def rectangular_mask(coords, img_dim):
    """

    :param coords:
    :param img_dim:
    :return:
    """
    coords = np.squeeze(coords)

    xpts = np.array(coords[:, 1])
    ypts = np.array(coords[:, 0])

    x_max = int(round(np.max(xpts)))
    x_min = int(round(np.min(xpts)))

    y_max = int(round(np.max(ypts)))
    y_min = int(round(np.min(ypts)))

    # Create the mask of the rect
    arr = np.zeros([img_dim[0], img_dim[1]])
    arr[x_min:x_max+1, y_min:y_max+1] = 1
    arr[arr == 0] = np.nan

    return arr
