import numpy as np

def split_picture(test_picture):
    split_1 = test_picture[:400, :400]
    split_2 = test_picture[:400, -400:]
    split_3 = test_picture[-400:, :400]
    split_4 = test_picture[-400:, -400:]
    return [split_1, split_2, split_3, split_4]

def merge_splits(split_1, split_2, split_3, split_4):
    intersect1 = np.mean(np.array([split_1[:208, 208:], split_2[:208, :192]]), axis=0)
    intersect2 = np.mean(np.array([split_3[192:, 208:], split_4[192:, :192]]), axis=0)

    intersect3 = np.mean(np.array([split_1[208:, :208], split_3[:192, :208]]), axis=0)
    intersect4 = np.mean(np.array([split_2[208:, 192:], split_4[:192, 192:]]), axis=0)

    intersect5 = np.mean(np.array([split_1[208:, 208:], 
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

def merge_predictions(predictions):
    merged = []

    for i in range(0, len(predictions), 4):
        merged.append(merge_splits(predictions[i], predictions[i+1], predictions[i+2], predictions[i+3]))

    return merged