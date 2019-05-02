
# Download data from the Kaggle Competition DSTL Satellite Images-> https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection #


import tifffile
import shapely.wkt as wkt
import pandas as pd
import cv2
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
from descartes.patch import PolygonPatch
from matplotlib.patches import Patch
import random
from matplotlib import cm
from shapely import affinity
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from collections import defaultdict
import sys
import seaborn as sns
import os
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, KFold
from keras.models import model_from_json


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Lambda, Activation,UpSampling2D
from keras.layers.merge import add,concatenate
from keras.optimizers import Nadam, Adam, TFOptimizer, SGD, RMSprop, Nadam
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D,GlobalAveragePooling2D
from keras.callbacks import Callback, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras.activations import sigmoid
from keras.losses import binary_crossentropy

from keras.utils.vis_utils import plot_model
#from keras_tqdm import TQDMNotebookCallback
from keras import backend as K

#from yellowfin import YFOptimizer
from tqdm import tqdm

CLASSES = {
    1: 'Bldg',
    2: 'Struct',
    3: 'Road',
    4: 'Track',
    5: 'Trees',
    6: 'Crops',
    7: 'Fast H2O',
    8: 'Slow H2O',
    9: 'Truck',
    10: 'Car',
}


data_dir = 'C:/Users/User1/Documents/Dstldata/'
# train_wkt_v4.csv stores the polygon data for all images and classes. The polygons
# uses relative coordinate positions.
_df = pd.read_csv(data_dir + 'train_wkt_v4.csv',
                  names=['ImageId', 'ClassId', 'MultipolygonWKT'], skiprows = 1)

# grid_sizes.csv stores the relative size of for each image. The origin is at the
# upper left corner, which means Xmax is positive and Ymin is negative.
_df1 = pd.read_csv(data_dir + 'grid_sizes.csv',
                   names = ['ImageId', 'Xmax', 'Ymin'], skiprows = 1)

# sample_submission.csv is the file for submission
_df2 = pd.read_csv(data_dir + 'sample_submission.csv',
                  names=['ImageId', 'ClassId', 'MultipolygonWKT'], skiprows = 1)

duplicates = []

train_wkt_v4 = _df[np.invert(np.in1d(_df.ImageId, duplicates))]
grid_sizes = _df1[np.invert(np.in1d(_df1.ImageId, duplicates))]
test_wkt = _df2

all_train_names = sorted(train_wkt_v4.ImageId.unique())
all_test_names = sorted(test_wkt.ImageId.unique())

train_IDs_dict = dict(zip(np.arange(len(all_train_names)), all_train_names))
train_IDs_dict_r = dict(zip(all_train_names, np.arange(len(all_train_names))))

test_IDs_dict = dict(zip(np.arange(len(all_test_names)), all_test_names))
test_IDs_dict_r = dict(zip(all_test_names, np.arange(len(all_test_names))))

x_crop = 3345
y_crop = 3338

test_names = ['6110_1_2', '6110_3_1', '6100_1_3', '6120_2_2']
train_names = all_train_names
test_ids = [train_IDs_dict_r[name] for name in test_names]
train_ids = [train_IDs_dict_r[name] for name in train_names]

#Reference : __author__ =rogerxujiang: dstl_unet

def resize(im, shape_out):
    '''
    Resize an image using cv2.
    Note: x and y are switched in cv2.resize
    :param im:
    :param shape_out:
    :return:
    '''
    return cv2.resize(im, (shape_out[1], shape_out[0]),
                      interpolation=cv2.INTER_CUBIC)



def affine_transform(img, warp_matrix, out_shape):
    '''
    Apply affine transformation using warp_matrix to img, and perform
    interpolation as needed
    :param img:
    :param warp_matrix:
    :param out_shape:
    :return:
    '''
    new_img = cv2.warpAffine(img, warp_matrix, (out_shape[1], out_shape[0]),
                             flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                             borderMode= cv2.BORDER_REPLICATE)
    # new_img[new_img == 0] = np.average(new_img)
    return new_img



def get_polygon_list(image_id, class_type):
    '''
    Load the wkt data (relative coordiantes of polygons) from csv file and
    returns a list of polygons (in the format of shapely multipolygon)
    :param image_id:
    :param class_type:
    :return:
    '''
    all_polygon = train_wkt_v4[train_wkt_v4.ImageId == image_id]
    polygon = all_polygon[all_polygon.ClassId == class_type].MultipolygonWKT
    # For empty polygon, polygon is a string of 'MULTIPOLYGON EMPTY'
    # wkt.loads will automatically handle this and len(polygon_list) returns 0
    # But polygon_list will never be None!
    polygon_list = wkt.loads(polygon.values[0])

    return polygon_list



def convert_coordinate_to_raster(coords, img_size, xymax):
    '''
    Converts the relative coordinates of contours into raster coordinates.
    :param coords:
    :param img_size:
    :param xymax:
    :return:
    '''
    xmax, ymax = xymax
    width, height = img_size

    coords[:, 0] *= (height + 1) / xmax
    coords[:, 1] *= (width + 1) / ymax

    coords = np.round(coords).astype(np.int32)

    return coords



def generate_contours(polygon_list, img_size, xymax):
    '''
    Convert shapely MultipolygonWKT type of data (relative coordinate) into
    list type of date for polygon raster coordinates
    :param polygon_list:
    :param img_size:
    :param xymax:
    :return:
    '''
    if len(polygon_list) == 0:
        return [], []

    to_ind = lambda x: np.array(list(x)).astype(np.float32)

    perim_list = [convert_coordinate_to_raster(to_ind(poly.exterior.coords),
                                               img_size, xymax)
                  for poly in polygon_list]
    inter_list = [convert_coordinate_to_raster(
        to_ind(poly.coords), img_size, xymax)
        for poly_ex in polygon_list for poly in poly_ex.interiors]

    return perim_list, inter_list

def image_stat(image_id):
    '''
    Return the statistics ofd an image as a pd dataframe
    :param image_id:
    :return:
    '''
    counts, total_area, mean_area, std_area = {}, {}, {}, {}
    img_area = get_image_area(image_id)

    for cl in CLASSES:
        polygon_list = get_polygon_list(image_id, cl)
        counts[cl] = len(polygon_list)
        if len(polygon_list) > 0:
            total_area[cl] = np.sum([poly.area for poly in polygon_list])\
                             / img_area * 100.
            mean_area[cl] = np.mean([poly.area for poly in polygon_list])\
                            / img_area * 100.
            std_area[cl] = np.std([poly.area for poly in polygon_list])\
                           / img_area * 100.

    return pd.DataFrame({'Class': CLASSES, 'Counts': counts,
                         'TotalArea': total_area, 'MeanArea': mean_area,
                         'STDArea': std_area})

def get_image_area(image_id):
    '''
    Calculate the area of an image
    :param image_id:
    :return:
    '''
    xmax = grid_sizes[grid_sizes.ImageId == image_id].Xmax.values[0]
    ymin = grid_sizes[grid_sizes.ImageId == image_id].Ymin.values[0]

    return abs(xmax * ymin)

def generate_mask_from_contours(img_size, perim_list, inter_list, class_id = 1):
    '''
    Create pixel-wise mask from contours from polygon of raster coordinates
    :param img_size:
    :param perim_list:
    :param inter_list:
    :param class_id:
    :return:
    '''
    mask = np.zeros(img_size, np.uint8)

    if perim_list is None:
        return mask
    # mask should match the dimension of image
    # however, cv2.fillpoly assumes the x and y axes are oppsite between mask and
    # perim_list (inter_list)
    cv2.fillPoly(mask, perim_list, class_id)
    cv2.fillPoly(mask, inter_list, 0)

    return mask


def scale_percentile(img):
    '''
    Scale an image's 1 - 99 percentiles into 0 - 1 for display
    :param img:
    :return:
    '''
    orig_shape = img.shape
    if len(orig_shape) == 3:
        img = np.reshape(img,
                         [orig_shape[0] * orig_shape[1], orig_shape[2]]
                         ).astype(np.float32)
    elif len(orig_shape) == 2:
        img = np.reshape(img, [orig_shape[0] * orig_shape[1]]).astype(np.float32)
    mins = np.percentile(img, 1, axis = 0)
    maxs = np.percentile(img, 99, axis = 0) - mins

    img = (img - mins) / maxs

    img.clip(0., 1.)
    img = np.reshape(img, orig_shape)

    return img



def rgb2gray(rgb):
    '''
    Converts rgb images to grey scale images
    :param rgb:
    :return:
    '''
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])



def crop(img, crop_coord):
    '''
    Crop out an patch from img, given the coordinates
    :param img:
    :param crop_coord:
    :return:
    '''
    width, height = img.shape[0], img.shape[1]
    x_lim = crop_coord[0].astype(np.int)
    y_lim = crop_coord[1].astype(np.int)

    assert 0 <= x_lim[0] < x_lim[1] <= width
    assert 0 <= y_lim[0] < y_lim[1] <= height

    return img[x_lim[0]: x_lim[1], y_lim[0]: y_lim[1]]


def calculate_class_weights():
    '''
    :return: class-wise true-label-area / false-label-area as a dictionary
    '''
    df = collect_stats()
    df = df.fillna(0)
    df = df.pivot(index = 'Class', columns = 'ImageId', values = 'TotalArea')
    df = df.sum(axis=1)
    df = df / (2500. - df)
    return df.to_dict()



def generate_train_ids(cl):
    '''
    Create train ids, and exclude the images with no true labels
    :param cl: 
    :return: 
    '''
    df = data_utils.collect_stats()
    df = df.pivot(index = 'ImageId', columns = 'Class', values = 'TotalArea')
    df = df.fillna(0)
    df = df[df[data_utils.CLASSES[cl + 1]] != 0]

    train_names = sorted(list(df.index.get_values()))

    return [data_utils.train_IDs_dict_r[name] for name in train_names]


def get_all_data(img_ids,train = True):
    '''
    Load all the training feature and label into memory. This requires 35 GB
    memory on Mac and takes a few minutes to finish.
    :return:
    '''
    print('Entered')
    image_feature = []
    image_label = []
    no_img = len(img_ids)
    phase = ['validation', 'training'][train]
    for i in range(no_img):
        id = img_ids[i]
        image_data = ImageData(id)
        
        image_data.create_train_feature()
        image_data.create_label()

        image_feature.append(image_data.train_feature[: x_crop, : y_crop, :])
        image_label.append(image_data.label[: x_crop, : y_crop, :])

        sys.stdout.write('\rLoading {} data: [{}{}] {}%\n'.\
                         format(phase,
                                '=' * i,
                                ' ' * (no_img - i - 1),
                                100 * i / (no_img - 1)))
        sys.stdout.flush()
    sys.stdout.write('\n')
    image_feature = np.stack(image_feature, -1)
    image_label = np.stack(image_label, -1)

    sys.stdout.write('Labels are{}valid.\n'.format(
        ' ' if np.isfinite(image_label).all() and \
               (image_label >= 0).all() and (image_label <= 1).all() else ' not '))
    sys.stdout.write('Image features are{}valid.\n'.format(
        ' ' if np.isfinite(image_feature).all() and \
               (image_feature >= -5000).all() and (image_feature <= 5000).all() \
            else ' not '))
    sys.stdout.write('\n')

    sys.stdout.flush()

    return np.rollaxis(image_feature, 3, 0), np.rollaxis(image_label, 3, 0)


def input_data(crop_size, class_id , crop_per_img ,
               reflection = True, rotation = 8, train = True, verbose = False):
    '''
    Returns the training images (feature) and the corresponding labels
    :param crop_size:
    :param class_id:
    :param crop_per_img:
    :param reflection:
    :param rotation:
    :param train:
    :return:
    '''

    # img_ids = generate_train_ids(class_id) if train else test_ids
    print('Entered')

    img_ids = train_ids if train else test_ids
    print(img_ids)
    no_img = len(img_ids)
    image_feature, image_label = get_all_data(img_ids,train = train)

    while True:

        images = []
        labels = []

        # Rotation angle is assumed to be the same, so that the
        # transformation only needs to be calculated once.
        if not rotation or rotation == 1:
            crop_diff = 0
            crop_size_new = crop_size

        else:
            angle = 360. * np.random.randint(0, rotation) / rotation
            radian = 2. * np.pi * angle / 360.
            if verbose:
                print ('Rotation angle : {0}(degree), {1: 0.2f}(radian)'.format(int(angle), radian))

            crop_size_new = int(
                np.ceil(float(crop_size) * (abs(np.sin(radian)) +
                                            abs(np.cos(radian)))))
            rot_mat = cv2.getRotationMatrix2D((float(crop_size_new) / 2.,
                                               float(crop_size_new) / 2.),
                                              angle, 1.)
            crop_diff = int((crop_size_new - crop_size) / 2.)

        np.random.shuffle(img_ids)

        for i in range(no_img):
            id = img_ids[i]
            for _ in range(crop_per_img):

                x_base = np.random.randint(0, x_crop - crop_size_new)
                y_base = np.random.randint(0, y_crop - crop_size_new)
                if verbose:
                    print('x_base {} for No. {} image'.format(x_base, id))
                    print('y_base {} for No. {} image'.format(y_base, id))

                img_crop = np.squeeze(image_feature[i, x_base: x_base + crop_size_new,y_base: y_base + crop_size_new, :])
                label_crop = np.squeeze(image_label[i, x_base: x_base + crop_size_new,
                                        y_base: y_base + crop_size_new, class_id])
                if not rotation or rotation == 1:
                    img_rot = img_crop
                    label_rot = label_crop
                else:
                    img_rot = cv2.warpAffine(img_crop, rot_mat,
                                             (crop_size_new, crop_size_new))
                    label_rot = cv2.warpAffine(label_crop, rot_mat,
                                               (crop_size_new, crop_size_new))

                x_step = 1 if not reflection else \
                    [-1, 1][np.random.randint(0, 2)]
                y_step = 1 if not reflection else \
                    [-1, 1][np.random.randint(0, 2)]

                images.append(img_rot[crop_diff: crop_diff + crop_size:,
                              crop_diff: crop_diff + crop_size, :]\
                                  [:: x_step, :: y_step, :])
                labels.append(label_rot[crop_diff: crop_diff + crop_size,
                              crop_diff: crop_diff + crop_size]\
                                  [:: x_step, :: y_step])

        yield np.stack(images, 0), np.stack(labels, 0)

def convert_image_to_display(img):
    '''
    
    :param img: 
    :return: 
    '''

    return 255 * normalize_image(img)



def calculate_warp_matrix(img1, img2, img_id):

    img_size = img1.shape
    img_size1 = img2.shape

    nx = img_size[0]
    ny = img_size[1]

    nx_1 = img_size1[0]
    ny_1 = img_size1[1]

    if [nx, ny] != [nx_1, ny_1]:
        img2 = resize(img2, [nx, ny])

    # Crop the center area to avoid the boundary effect.
    p1 = img1[int(nx * 0.2):int(nx * 0.8), int(ny * 0.2):int(ny * 0.8), :].astype(np.float32)
    p2 = img2[int(nx * 0.2):int(nx * 0.8), int(ny * 0.2):int(ny * 0.8), :].astype(np.float32)

    p1 = normalize_image(p1)
    p2 = normalize_image(p2)

    p1 = np.sum(p1, axis=2)
    p2 = np.sum(p2, axis=2)

    warp_mode = cv2.MOTION_EUCLIDEAN
    warp_mat = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-7)

    try:
        (cc, warp_mat) = cv2.findTransformECC(p1, p2, warp_mat, warp_mode, criteria)
    except:
        warp_mat = np.eye(2, 3, dtype=np.float32)
        cc = 0.

    print("Align two rasters for {}: cc:{}".format(img_id, cc))
    print("warp matrix: ")
    print(warp_mat)

    return warp_mat



def normalize_image(img):
    '''
    Normalize the image (10% to 90%) to [0,1]
    :param img: 
    :return: 
    '''
    img_shape = img.shape

    if len(img_shape) == 3:
        for channel in range(img_shape[2]):
            im_min = np.percentile(img[:,:, channel], 10)
            im_max = np.percentile(img[:,:, channel], 90)
            img[:,:, channel] = (img[:,:, channel].astype(np.float32) - im_min) / (im_max - im_min)
    elif len(img_shape) == 2:
        im_min = np.percentile(img, 10)
        im_max = np.percentile(img, 90)
        img = (img.astype(np.float32) - im_min) / (im_max - im_min)

    return img

class ImageData():

    def __init__(self,image_id, phase='train'):

        self.image_id = train_IDs_dict[image_id] \
            if phase == 'train' else test_IDs_dict[image_id]
        self.stat = image_stat(self.image_id) if phase == 'train' else None
        self.three_band_image = None
        self.sixteen_band_image = None
        self.image = None
        self.image_size = None
        self._xymax = None
        self.label = None
        self.crop_image = None
        self.train_feature = None
        self.pred_mask = None
        print('ok')

    def load_pre_mask(self):
        self.pred_mask = None


    def load_image(self):
        '''
        Load three band and sixteen band images, registered and at the same
        resolution
        Assign value for image_size
        :return:
        '''
        im = self.image_stack()
        self.three_band_image = im[..., 0: 3]
        self.sixteen_band_image = im[..., 3:]
        self.image = im
        self.image_size = np.shape(im)[0: 2]
        xmax = grid_sizes[grid_sizes.ImageId == self.image_id].Xmax.values[0]
        ymax = grid_sizes[grid_sizes.ImageId == self.image_id].Ymin.values[0]
        self._xymax = [xmax, ymax]


    def get_image_path(self):
        '''
        Returns the paths for all images
        :return:
        '''
        return {
            '3': '{}three_band/{}.tif'.format(data_dir, self.image_id),
            'A': '{}sixteen_band/{}_A.tif'.format(data_dir, self.image_id),
            'M': '{}sixteen_band/{}_M.tif'.format(data_dir, self.image_id),
            'P': '{}sixteen_band/{}_P.tif'.format(data_dir, self.image_id)
        }


    def read_image(self):
        '''
        Read all original images
        :return:
        '''
        images = {}
        path = self.get_image_path()

        for key in path:
            im = tifffile.imread(path[key])
            if key != 'P':
                images[key] = np.transpose(im, (1, 2, 0))
            elif key == 'P':
                images[key] = im

        im3 = images['3']
        ima = images['A']
        imm = images['M']
        imp = images['P']

        [nx, ny, _] = im3.shape

        images['A'] = resize(ima, [nx, ny])
        images['M'] = resize(imm, [nx, ny])
        images['P'] = resize(imp, [nx, ny])

        return images


    def image_stack(self):
        '''
        Resample all images to highest resolution and align all images
        :return:
        '''

        images = self.read_image()

        im3 = images['3']
        ima = images['A']
        imm = images['M']
        imp = images['P']

        imp = np.expand_dims(imp, 2)

        [nx, ny, _] = im3.shape

        warp_matrix_a = np.load(
            (data_dir +
             'image_alignment/{}_warp_matrix_a.npz').format(self.image_id)
        )
        warp_matrix_m = np.load(
            (data_dir +
             'image_alignment/{}_warp_matrix_m.npz').format(self.image_id)
        )

        ima = affine_transform(ima, warp_matrix_a, [nx, ny])
        imm = affine_transform(imm, warp_matrix_m, [nx, ny])

        im = np.concatenate((im3, ima, imm, imp), axis = -1)

        return im


    def create_label(self):
        '''
        Create the class labels
        :return:
        '''
        if self.image is None:
            self.load_image()
        labels = np.zeros(np.append(self.image_size, len(CLASSES)), np.uint8)

        for cl in CLASSES:
            polygon_list = get_polygon_list(self.image_id, cl)
            perim_list, inter_list = generate_contours(
                polygon_list, self.image_size, self._xymax)
            mask = generate_mask_from_contours(
                self.image_size, perim_list, inter_list, class_id = 1)
            labels[..., cl - 1] = mask
        self.label = labels


    def create_train_feature(self):
        '''
        Create synthesized features
        :return:
        '''
        if self.three_band_image is None:
            self.load_image()

        m = self.sixteen_band_image[..., 8:].astype(np.float32)
        rgb = self.three_band_image.astype(np.float32)

        image_r = rgb[..., 0]
        image_g = rgb[..., 1]
        image_b = rgb[..., 2]

        nir = m[..., 7]
        re = m[..., 5]


        ndwi = (image_g - nir) / (image_g + nir)
        ndwi = np.expand_dims(ndwi, 2)


        # binary = (ccci > 0.11).astype(np.float32) marks water fairly well
        ccci = np.nan_to_num(
            (nir - re) / (nir + re) * (nir - image_r) / (nir + image_r))
        ccci = ccci.clip(
            max=np.percentile(ccci, 99.9),
            min=np.percentile(ccci, 0.1))
        ccci = np.expand_dims(ccci, 2)

#       feature = np.concatenate([m, rgb, evi, ndwi, savi, ccci], 2)
        feature = np.concatenate([ rgb], 2)
        
        #print('ndwiiiiii',feature.shape)
        #feature = rgb
        feature[feature == np.inf] = 0
        feature[feature == -np.inf] = 0

        self.train_feature = feature


    def apply_crop(self, patch_size, ref_point = [0, 0], method = 'random'):

        if self.image is None:
            self.load_image()

        crop_area = np.zeros([2, 2])
        width = self.image_size[0]
        height = self.image_size[1]

        assert width >= patch_size > 0 and patch_size <= height

        if method == 'random':
            ref_point[0] = random.randint(0, width - patch_size)
            ref_point[1] = random.randint(0, height - patch_size)
            crop_area[0][0] = ref_point[0]
            crop_area[1][0] = ref_point[1]
            crop_area[0][1] = ref_point[0] + patch_size
            crop_area[1][1] = ref_point[1] + patch_size
        elif method == 'grid':
            assert width > ref_point[0] + patch_size
            assert height > ref_point[1] + patch_size
            crop_area[0][0] = ref_point[0]
            crop_area[1][0] = ref_point[1]
            crop_area[0][1] = ref_point[0] + patch_size
            crop_area[1][1] = ref_point[1] + patch_size
        else:
            raise NotImplementedError(
                '"method" should either be "random" or "grid"')
        self.crop_image = crop(self.image, crop_area)




#Reference: https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/

def bn(x):
    return BatchNormalization()(x)
  
def pool2d(x):
    return MaxPooling2D(pool_size=(2,2),padding='same')(x)
  
def act(x):
    return Activation('relu')(x)
  
def up2d(x):
    return UpSampling2D(size=(2,2))(x)
  
def drp(x,p):
    return Dropout(1-p)(x) if p else x

def conv_block_1(x,nf,size,stride,act='relu'):
    x = Conv2D(nf,size,strides=(stride,stride),padding='same',activation=act)(x)
    return x
  
def conv_block_2(x,nf1,nf2,size,stride,act='relu'):
    x = Conv2D(nf1,size,strides=(stride,stride),padding='same',activation=act)(x)
    x = Conv2D(nf2,size,strides=(stride,stride),padding='same',activation=act)(x)
    return x
  
def conv_block_3(x,nf,size,stride,act='relu'):
    x = Conv2D(nf,size,strides=(stride,stride),padding='same',activation=act)(x)
    x = Conv2D(nf,size,strides=(stride,stride),padding='same',activation=act)(x)
    x = Conv2D(nf,size,strides=(stride,stride),padding='same',activation=act)(x)
    return x
  
def mod_unet(arr,nf=64,nf2=96):
    inp = Input(shape=arr.shape[1:])
    print(inp.shape)
    
    conv1 = conv_block_1(inp,nf,3,1) # 64x256x256
    print(conv1.shape)
    
    conv2 = conv_block_2(bn(conv1),nf,nf,3,1)
    pool2 = pool2d(conv2) # 64x128x128
    print(pool2.shape)
    
    conv3 = conv_block_3(bn(pool2),nf,3,1)
    pool3 = pool2d(conv3) # 64x64x64
    print(pool3.shape)
    
    conv4 = conv_block_3(bn(pool3),nf,3,1)
    pool4 = pool2d(conv4) # 64x32x32
    print(pool4.shape)
   
    conv5 = conv_block_3(bn(pool4),nf,3,1)
    pool5 = pool2d(conv5) # 64x16x16
    print(pool5.shape)
    
    conv6 = conv_block_3(bn(pool5),nf,3,1)
    pool6 = pool2d(conv6) # 64x8x8
    print(pool6.shape)
    
    conv7 = conv_block_2(bn(pool6),nf,nf,3,1)
    up7 = up2d(conv7)
    up7 = conv_block_1(bn(up7),nf,3,1) # 64x16x16
    print(up7.shape)
    
    up7 = concatenate([up7,conv6],axis=-1)
    conv8 = conv_block_2(bn(up7),nf2,nf,3,1)
    up8 = up2d(conv8)
    up8 = conv_block_1(bn(up8),nf,3,1) # 64x32x32
    print(up8.shape)
    
    up8 = concatenate([up8,conv5],axis=-1)
    conv9 = conv_block_2(bn(up8),nf2,nf,3,1)
    up9 = up2d(conv9)
    up9 = conv_block_1(bn(up9),nf,3,1) # 64x64x64
    print(up9.shape)
    
    up9 = concatenate([up9,conv4],axis=-1)
    conv10 = conv_block_2(bn(up9),nf2,nf,3,1)
    up10 = up2d(conv10)
    up10 = conv_block_1(bn(up10),nf,3,1) # 64x128x128
    print(up10.shape)

    up10 = concatenate([up10,conv3],axis=-1)
    conv11 = conv_block_2(bn(up10),nf2,nf,3,1)
    up11 = up2d(conv11)
    up11 = conv_block_1(bn(up11),nf,3,1) # 64x256x256
    print(up11.shape)

    up11 = concatenate([up11,conv2],axis=-1)
    conv12 = conv_block_2(bn(up11),nf2,nf,3,1)
    conv_out = drp(conv_block_1(conv12,1,1,1,act='sigmoid'),0.5) 
    #seg_out = sigmoid(conv_out) # 1x256x256
    print(conv_out.shape)
    
    return inp, conv_out

smooth = 1e-12
def jaccard_approx(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

  
def loss_func(y_true,y_pred):
    
    crossentropy_loss = binary_crossentropy(y_true,y_pred)

    
    return crossentropy_loss - K.log(jaccard_approx(y_true,y_pred))


#x_train = np.full((1,144,144,3), 0)
#inp,outp = mod_unet(x_train)
#unet_m = Model(inp,outp)


train_input = input_data( class_id=8, crop_size=256, crop_per_img=20, rotation=360, verbose=True, train=True)
test_output = input_data( class_id=7, crop_size=256, crop_per_img=50, rotation=360, verbose=True, train=False)

img, label = next(train_input)
label = label[:, :, :,newaxis]

'''
ind = 0
fig, axs = plt.subplots(15,5, figsize=[10,10])
for i in range(15):
    for j in range(5):
        axs[i,j].imshow(scale_percentile(img[ind, :,:,:3]))
        ind = ind+1
plt.show()

ind = 0
fig, axs = plt.subplots(15,5, figsize=[10,10])
for i in range(15):
    for j in range(5):
        axs[i,j].imshow(scale_percentile(label[ind, :,:,0]), cmap=plt.cm.gray)
        ind = ind+1
plt.show()
'''
test_img,label2 = next(test_output)
'''
ind = 0
fig, axs = plt.subplots(8,10, figsize=[20,20])
for i in range(8):
    for j in range(10):
        axs[i,j].imshow(scale_percentile(test_img[ind, :,:,:3]))
        ind = ind+1
plt.show()


print(img.shape), print(label.shape)
print(test_img.shape)
'''
'''
ind = 0
fig, axs = plt.subplots(5,5, figsize=[20,20])
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(scale_percentile(img[ind, :,:,:]))
        ind = ind+1
plt.show()

ind = 0
fig, axs = plt.subplots(5,5, figsize=[20,20])
for i in range(5):
    for j in range(5):
        axs[i,j].imshow(label[ind, :, :], cmap=plt.cm.gray)
        ind = ind+1
plt.show()
'''

inp1,outp1 = mod_unet(img)
unet_m = Model(inp1,outp1)

opt = Nadam()
# opt = TFOptimizer(YFOptimizer())
unet_m.compile(loss=loss_func,optimizer=opt,metrics=['accuracy',jaccard_approx])
# K.set_value(unet_m.optimizer.learn_rate,1)
history = unet_m.fit(img,label,validation_split=0.33,batch_size=2,epochs=3)

print(history.history.keys())
# summarize history for accuracy

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy-Large vehicles')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss-Large Vehicles')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



# serialize model to JSON
model_json = unet_m.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
unet_m.save_weights("model2.h5")
print("Saved model to disk")

# Uncomment below lines after saving the model
'''
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
unet_m = model_from_json(loaded_model_json)
# load weights into new model
unet_m.load_weights("model2.h5")
print("Loaded model from disk")

#print(test_img.shape)

unet_m_predictions = unet_m.predict(test_img)
print(unet_m_predictions.shape)


for i in range(0,200,20):
    ind = i
    fig, axs = plt.subplots(5,4, figsize=[20,20])
    fig2, axs2 = plt.subplots(5,4, figsize=[20,20])

    for i in range(5):
        for j in range(4):
            axs[i,j].imshow(scale_percentile(test_img[ind, :,:,:]))
            axs2[i,j].imshow(scale_percentile(unet_m_predictions[ind, :, :,0]), cmap=plt.cm.gray)
            ind = ind+1

    plt.show()
'''