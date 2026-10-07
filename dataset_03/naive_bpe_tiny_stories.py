# install gdown

import gdown

# Replace with your actual public Google Drive link
url = "https://drive.google.com/file/d/1UL5hkH-FFhevASS3k6d7qoDGlfuZyP1s/view?usp=sharing"

custom_name = 'dataset_03/TinyStories-train.bin'

# Downloads and renames the file
gdown.download(url, output=custom_name, quiet=False)

