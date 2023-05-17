import os
from PIL import Image
path = "./img"
files =os.listdir(path)

for data in files:
    path_img = path + '/' + data;
    img_file = os.listdir(path_img)
    for img_file_name in img_file:
        path_file_img = path_img  + '/' + img_file_name
        print(path_file_img)
        img = Image.open(path_file_img)
        img = img.convert("RGB")
        img.save(path_file_img)