import math
import numpy as np


def split_picture(test_picture):
    split_1 = test_picture[:400, :400]
    split_2 = test_picture[:400, -400:]
    split_3 = test_picture[-400:, :400]
    split_4 = test_picture[-400:, -400:]
    return [split_1, split_2, split_3, split_4]


def merge_splits(split_1, split_2, split_3, split_4, mode='mean'):
    assert mode in ['mean', 'max'], 'mode can only be one between mean and max!'
    if mode == 'mean':
        function = np.mean
    elif mode == 'max':
        function = np.max
    intersect1 = function(np.array([split_1[:208, 208:], split_2[:208, :192]]), axis=0)
    intersect2 = function(np.array([split_3[192:, 208:], split_4[192:, :192]]), axis=0)

    intersect3 = function(np.array([split_1[208:, :208], split_3[:192, :208]]), axis=0)
    intersect4 = function(np.array([split_2[208:, 192:], split_4[:192, 192:]]), axis=0)

    intersect5 = function(np.array([split_1[208:, 208:], 
                                   split_2[208:, :192],
                                   split_3[:192, 208:],
                                   split_4[:192, :192]]), axis=0)

    north_west = split_1[:208, :208]
    north_east = split_2[:208, 192:]
    south_west = split_3[192:, :208]
    south_east = split_4[192:, 192:]

    upper_slice = np.concatenate([north_west, intersect1, north_east], axis= 1)
    middle_slice = np.concatenate([intersect3, intersect5, intersect4], axis= 1)
    lower_slice = np.concatenate([south_west, intersect2, south_east], axis= 1)

    final_image = np.concatenate([upper_slice, middle_slice, lower_slice], axis= 0)
    return final_image


def preprocess_test_images(test_images):
    preprocessed = []

    for image in test_images:
        preprocessed.extend(split_picture(image))

    return np.array(preprocessed)


def merge_predictions(predictions, mode='mean'):
    merged = []

    for i in range(0, len(predictions), 4):
        merged.append(merge_splits(predictions[i], predictions[i+1], predictions[i+2], predictions[i+3], mode))

    return merged


def single_model_training(model, save_path, additional_epochs=30, competition_epochs=60, b_size=8, loss='dice', l_rate_a=.0001, l_rate_b=.00001):
    print('Training model {}.\nParameters:'.format(model.net_type))
    print('\tbatch_size: {};\n\tloss: {};\n\tl_rate_google_data: {};\n\tl_rate_competition_data: {};'.format(b_size, loss, l_rate_a, l_rate_b))
    print('Training on additional data.')
    model.train(loss=loss, epochs=additional_epochs, train_on='google_data', l_rate=l_rate_a, batch_size=b_size)
    print('Training on competition data.')
    model.train(loss=loss, epochs=competition_epochs, train_on='competition_data', l_rate=l_rate_b, batch_size=b_size)
    print('Saving model at path: {}'.format(save_path))
    model.save_model(save_path)
    return

# the code below was rearranged by the code provided by the course's TAs
# and it's used to build masks from a submission csv file


def binary_to_uint8(img):
    rimg = (img * 255).round().astype(np.uint8)
    return rimg


def reconstruct_from_labels(lines, image_id):
    h = 16
    w = h
    imgwidth = int(math.ceil((600.0/w))*w)
    imgheight = int(math.ceil((600.0/h))*h)
    nc = 3
    
    im = np.zeros((imgwidth, imgheight), dtype=np.uint8)
    image_id_str = '%.3d_' % image_id
    for i in range(1, len(lines)):
        line = lines[i]
        if not image_id_str in line:
            continue

        tokens = line.split(',')
        id = tokens[0]
        prediction = int(tokens[1])
        tokens = id.split('_')
        i = int(tokens[1])
        j = int(tokens[2])

        je = min(j+w, imgwidth)
        ie = min(i+h, imgheight)
        if prediction == 0:
            adata = np.zeros((w,h))
        else:
            adata = np.ones((w,h))

        im[j:je, i:ie] = binary_to_uint8(adata)
        
    return im


def submission_outputs(label_file, numbers):
    f = open(label_file)
    lines = f.readlines()
    
    predictions = []
    for num in numbers:
        predictions.append(reconstruct_from_labels(lines, num))
    
    return predictions
