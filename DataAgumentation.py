from keras.preprocessing.image import ImageDataGenerator
import os
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
data_dir = r'datasets\animals'

for subdir in os.listdir(data_dir):
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path):
        file_list = os.listdir(subdir_path)
        print(f"Processing {len(file_list)} files in {subdir_path}")
        if len(file_list) < 1501:
            num_augmentations = 1501 - len(file_list)
            i = 0
            for batch in datagen.flow_from_directory(subdir_path,
                                                     batch_size=1,
                                                     save_to_dir=subdir_path,
                                                     save_prefix=f'{subdir}_aug_',
                                                     save_format='jpg',
        											 class_mode='categorical'
                                                     ):
                i += 1
                if i >= num_augmentations:
                    break

