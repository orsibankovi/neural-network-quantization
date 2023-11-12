import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import natsort

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class GetCOCODataset(Dataset):
    def __init__(self, dataDir, dataType):
        super(GetCOCODataset, self).__init__()
        self.dataDir = dataDir
        self.dataType = dataType
        self.annFile = os.path.join(self.dataDir, 'annotations', f'instances_{self.dataType}.json')
        self.coco = COCO(self.annFile)

        self.image_ids = list(self.coco.getImgIds())
        self.image_ids = natsort.natsorted(self.image_ids)

    def __getitem__(self, index):
        img_info = self.coco.loadImgs(self.image_ids[index])[0]
        img_path = os.path.join(self.dataDir, self.dataType, img_info['file_name'])
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_tensor = preprocess(image).float()

        ann_ids = self.coco.getAnnIds(imgIds=self.image_ids[index])
        annotations = self.coco.loadAnns(ann_ids)
        labels = [ann['category_id'] for ann in annotations]
        boxes = [ann['bbox'] for ann in annotations]
        labels_dict = {
            "labels": torch.tensor(labels).long(),
            "boxes": torch.tensor(boxes).float()
        }
        return img_path, image_tensor, labels_dict

    def __len__(self):
        return len(self.image_ids)
    
    def collate_fn(batch):
        paths, imgs, labels = zip(*batch)
        return paths, torch.stack(imgs, 0), labels