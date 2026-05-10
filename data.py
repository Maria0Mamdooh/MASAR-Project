from google.colab import drive
import zipfile


drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/AI System Design/University-Release.zip'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/dataset')

print("✅Unzipped!")


  import os

base_path = "/content/dataset"

for root, dirs, files in os.walk(base_path):
    level = root.replace(base_path, "").count(os.sep)
    if level < 3:
        print(root, "→", len(files), "files")


  import os
import glob
from PIL import Image
from torch.utils.data import Dataset

DATA_ROOT = "/content/dataset/University-Release"

TRAIN_DRONE = f"{DATA_ROOT}/train/drone"
TRAIN_SAT = f"{DATA_ROOT}/train/google"

TEST_QUERY = f"{DATA_ROOT}/test/query_drone"
TEST_GALLERY = f"{DATA_ROOT}/test/gallery_satellite"

for path in [TRAIN_DRONE, TRAIN_SAT, TEST_QUERY, TEST_GALLERY]:
    print(path, "exists:", os.path.exists(path))

  
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_first_image(folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    if len(files) == 0:
        return None
    return files[0]


# Dataset class for training: pairs drone and satellite images by building ID
class UniversityPairDataset(Dataset):
    def __init__(self, drone_root, sat_root, transform=None, max_ids=50):
        self.drone_root = drone_root
        self.sat_root = sat_root
        self.transform = transform
        self.samples = []

        drone_ids = set(os.listdir(drone_root))
        sat_ids = set(os.listdir(sat_root))

        common_ids = sorted(list(drone_ids.intersection(sat_ids)))

        for building_id in common_ids:
            drone_folder = os.path.join(drone_root, building_id)
            sat_folder = os.path.join(sat_root, building_id)

            if not os.path.isdir(drone_folder) or not os.path.isdir(sat_folder):
                continue

            drone_img = get_first_image(drone_folder)
            sat_img = get_first_image(sat_folder)

            if drone_img is not None and sat_img is not None:
                self.samples.append((drone_img, sat_img, building_id))

            if len(self.samples) >= max_ids:
                break

        print("Total paired samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        drone_path, sat_path, building_id = self.samples[index]

        drone_img = Image.open(drone_path).convert("RGB")
        sat_img = Image.open(sat_path).convert("RGB")

        if self.transform:
            drone_img = self.transform(drone_img)
            sat_img = self.transform(sat_img)

        return drone_img, sat_img, building_id



# Dataset class for evaluation: loads individual images (query or gallery)
class UniversityImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_ids=100):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        building_ids = sorted(os.listdir(root_dir))

        for building_id in building_ids:
            folder = os.path.join(root_dir, building_id)

            if not os.path.isdir(folder):
                continue

            img_path = get_first_image(folder)

            if img_path is not None:
                self.samples.append((img_path, building_id))

            if len(self.samples) >= max_ids:
                break

        print("Total images:", len(self.samples), "from", root_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, building_id = self.samples[index]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, building_id, img_path



train_dataset = UniversityPairDataset(
    drone_root=TRAIN_DRONE,
    sat_root=TRAIN_SAT,
    transform=image_transform,
    max_ids=200
)

sample_drone, sample_sat, sample_id = train_dataset[0]

print("Drone shape:", sample_drone.shape)
print("Satellite shape:", sample_sat.shape)
print("Building ID:", sample_id)



query_dataset = UniversityImageDataset(
    root_dir=TEST_QUERY,
    transform=image_transform,
    max_ids=100
)

gallery_dataset = UniversityImageDataset(
    root_dir=TEST_GALLERY,
    transform=image_transform,
    max_ids=100
)  

  

  
