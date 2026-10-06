# install gdown
import gdown

# check if gdown installed
try:
    import gdown
    print("gdown is already installed!")
except ImportError:
    print("gdown not found. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown
    print("gdown successfully installed and imported!")


# Replace with your actual public Google Drive link
url = "https://drive.google.com/file/d/1UL5hkH-FFhevASS3k6d7qoDGlfuZyP1s/view?usp=sharing"

# Path where the .bin file will be saved
output = "model.bin"

# Download the file
gdown.download(url, output, fuzzy=True)