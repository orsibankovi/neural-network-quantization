import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class GetDataset(Dataset):
    def __init__(self):
        super(GetDataset, self).__init__()
        self.path_input = "C:/Users/orsolya.bankovi/Documents/Uni/deepLearning_project/imagenet"
        self.input = os.listdir(self.path_input)
        self.input_names = list(filter(lambda x: x.endswith(".png"), list(self.input)))
        self.sorted_input = natsorted(self.input_names)
        self.input_images = []
        for count, images in enumerate(self.sorted_input[:]):
            input_image = Image.open(self.path_input + '/' + images)
            input_tensor = preprocess(input_image).float()
            self.input_images.append(input_tensor)
            
        self.labels = []
        with open("C:/Users/orsolya.bankovi/Documents/Uni/deepLearning_project/imagenet/labels.txt", 'r') as f:
            for i in range(1000):
                line = f.readline()
                self.labels.append(int(line))

    def __getitem__(self, index):
        return self.input_images[index], self.labels[index]

    def __len__(self):
        return len(self.input_images)