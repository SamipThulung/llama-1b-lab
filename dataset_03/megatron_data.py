# install gdown

import gdown

# Replace with your actual public Google Drive link
url1 = "https://drive.google.com/file/d/1O1D4Ikev4aVkDnNcdv5tPPuekwamYEqB/view?usp=sharing"
url2 = "https://drive.google.com/file/d/1Z6B1n1RA6qiRQnqaAU5PiO7WJgiZweQq/view?usp=sharing"

custom_name1 = 'megatron_07/megatron_2b.bin'
custom_name2 = 'megatron_07/megatron_2b.idx'

# Downloads and renames the file
gdown.download(url1, output=custom_name1, quiet=False)
gdown.download(url2, output=custom_name2, quiet=False)

