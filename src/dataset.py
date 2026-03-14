import cv2
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
import pprint


class FootballDataset(Dataset):
    def __init__(self, root, transform=None):
        self.matches = os.listdir(root)
        self.match_files = [os.path.join(root, match_file) for match_file in self.matches]

        # Use image id in json file to count
        self.from_id = 0
        self.to_id = 0

        # Use image_id to select video
        self.video_select = {}

        # Count total frame in all video
        for path in self.match_files:     # dataset có 3 video => duyệt từng video
            # Extract json file
            json_dir, video_dir = os.listdir(path)
            json_dir, video_dir = os.path.join(path, json_dir), os.path.join(path, video_dir)
            with open(json_dir, "r") as json_file:
                json_data = json.load(json_file)

            self.to_id += len(json_data["images"])
            self.video_select[path] = [self.from_id + 1, self.to_id]    # 3 video, mỗi video có 1500 frame
                                                                        # video 1 => lấy từ id 1 đến 1500,
                                                                        # video 2 => lấy từ id 1501 đến 3000,
                                                                        # video 3 => lấy từ id 3001 đến 4500
            self.from_id = self.to_id
        self.transform = transform

    def __len__(self):
        return self.to_id

    def __getitem__(self, idx):     # idx: 0 -> 4500
        # Choose real index and video of this frame
        for key, value in self.video_select.items():
            if value[0] <= idx + 1 <= value[1]:
                idx = idx - value[0]     # lưu ý: nếu thay phép - thành phép % thì sẽ đúng vs video 2 và 3 nhưng sai vs video 1
                select_path = key

        # Load file
        json_dir, video_dir = os.listdir(select_path)
        json_dir, video_dir = os.path.join(select_path, json_dir), os.path.join(select_path, video_dir)
        json_file = open(json_dir, "r")
        annotations = json.load(json_file)["annotations"]

        # pprint.pprint(annotations)

        # Real frame
        cap = cv2.VideoCapture(video_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        flag, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Take annotations
        annotations = [anno for anno in annotations if anno["image_id"] == idx + 1 and anno["category_id"] == 4]
        box = [annotation["bbox"] for annotation in annotations]
        cropped_images = [frame[int(y):int(y + h), int(x):int(x + w)] for [x, y, w, h] in box]

        # Take number of players
        jerseys = [int(annotation["attributes"]["jersey_number"]) for annotation in annotations]

        if self.transform:
            cropped_images = [self.transform(image) for image in cropped_images]
            cropped_images = torch.stack(cropped_images)

        return cropped_images, jerseys

def my_collate_fn(batch):
    images, labels = list(zip(*batch))
    images = torch.cat(images, dim=0)
    final_labels = []
    for label in labels:
        final_labels.extend(label)
    return images, torch.LongTensor(final_labels)

if __name__ == "__main__":
    transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        ToTensor(),
    ])
    index = 800
    dataset = FootballDataset("./data/football", transform=transform)
    cropped_images, jerseys = dataset.__getitem__(index)
    print(cropped_images)
    print(jerseys)
    print(cropped_images.shape)
    for image in cropped_images:
        print(image.shape)
    for image, jersey in zip(cropped_images, jerseys):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, (224, 448))
        cv2.imshow(str(jersey), image)
        cv2.waitKey(0)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=my_collate_fn,
    )
