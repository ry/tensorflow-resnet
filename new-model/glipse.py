


def one_hot_to_unit_value(x):
    x_index = tf.argmax(x, 0)
    return x_index / 127.0

def glimpse(hd_image, y, x, width):
    # let's have y and x and width be 1 hot vectors with 128 dimensions

    y_val = one_hot_to_unit_value(y)
    x_val = one_hot_to_unit_value(x)
    width_val = one_hot_to_unit_value(width)

    img_shape = hd_image.get_shape().as_list()
    img_height = img_shape[1]
    img_width = img_shape[2]

    glimpse_y = tf.to_int32(img_height * y_val)
    glimpse_x = tf.to_int32(img_width * x_val)
    
    # width of 1.0 means the whole image if centered at origin.
    pixel_width = 
    glimpse_x = tf.to_int32(img_width * x_val)




