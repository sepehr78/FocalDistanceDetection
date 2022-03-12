import torchvision.io
from torch.utils.data import Dataset, DataLoader


class VideoDataset(Dataset):
    def __init__(self, video_path, label, is_grayscale=True):
        self.label = label
        self.video_path = video_path
        self.video_tensor, _, _ = torchvision.io.read_video(video_path)
        if is_grayscale:
            self.video_tensor = self.video_tensor[:, :, :, 0].unsqueeze(1)
        self.video_tensor = 1.0 * self.video_tensor
        self.video_tensor = (self.video_tensor - self.video_tensor.min()) / (
                    self.video_tensor.max() - self.video_tensor.min())

    def __len__(self):
        return len(self.video_tensor)

    def __getitem__(self, item):
        return self.video_tensor[item], self.label
