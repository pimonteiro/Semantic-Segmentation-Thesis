import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


class SegmentationGenerator(Sequence):
    
    def __init__(self, dataset, mode='train', n_classes=19, batch_size=1, resize_shape=(512,512), 
                 seed = 7, horizontal_flip=False, blur = 0,
                 vertical_flip=0, brightness=0.1, rotation=0, zoom=0, do_ahisteq = False):
        
        self.blur = blur
        self.histeq = do_ahisteq
        self.dataset = pd.read_csv(dataset)
        self.dataset = self.dataset[dataset['subset'] == mode]

        
        np.random.seed(seed)
        
        self.mode = mode
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom

        self.image_path_list = sorted(self.dataset.x.tolist())
        self.label_path_list = sorted(self.dataset.y.tolist())

        if self.resize_shape:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, resize_shape[1]*resize_shape[0], 1), dtype='uint8')
        else:
            raise Exception('No image dimensions specified!')
        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size
        
    def __getitem__(self, i):
        
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], 
                                                        self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):
            
            image = cv2.imread(image_path, 1)
            label = cv2.imread(label_path, 0)
            
            if self.blur and random.randint(0,1):
                image = cv2.GaussianBlur(image, (self.blur, self.blur), 0)

            if self.resize_shape:
                image = cv2.resize(image, self.resize_shape)
                label = cv2.resize(label, self.resize_shape, interpolation = cv2.INTER_NEAREST)
        
            # Do augmentation
            if self.horizontal_flip and random.randint(0,1):
                image = cv2.flip(image, 1)
                label = cv2.flip(label, 1)
            if self.vertical_flip and random.randint(0,1):
                image = cv2.flip(image, 0)
                label = cv2.flip(label, 0)
            if self.brightness:
                factor = 1.0 + random.gauss(mu=0.0, sigma=self.brightness)
                if random.randint(0,1):
                    factor = 1.0/factor
                table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
                image = cv2.LUT(image, table)
            if self.rotation:
                angle = random.gauss(mu=0.0, sigma=self.rotation)
            else:
                angle = 0.0
            if self.zoom:
                scale = random.gauss(mu=1.0, sigma=self.zoom)
            else:
                scale = 1.0
            if self.rotation or self.zoom:
                M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                label = cv2.warpAffine(label, M, (label.shape[1], label.shape[0]))

            if self.histeq: # and convert to RGB
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) # to BGR
                 
            label = label.astype('uint8')
            self.X[n] = image  
            self.Y[n] = label

        return self.X, self.Y
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_list, self.label_path_list))
        random.shuffle(c)
        self.image_path_list, self.label_path_list = zip(*c)
                
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        image = cv2.resize(image, crop_shape)
        label = cv2.resize(label, crop_shape, interpolation = cv2.INTER_NEAREST)
        return image, label
