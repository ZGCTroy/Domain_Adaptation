import numpy as np
from torch.utils import data
from torchvision.datasets.utils import download_url
import os
from PIL import Image


class USPSDataset(data.Dataset):
    num_labels = 10
    image_shape = [16, 16, 1]

    urls = [
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz',
        'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz'
    ]
    training_file = 'zip.train.gz'
    test_file = 'zip.test.gz'


    def __init__(self, root, split='train', transform=None,
                 label_transform=None, download=True):

        super().__init__()

        self.root = root
        self.which = split

        self.transform = transform
        self.label_transform = label_transform

        if download:
            self.download()

        self.get_data(self.which)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        x = Image.fromarray(self.images[index])
        y = int(self.labels[index])

        if self.transform is not None:
            x = self.transform(x)

        if self.label_transform is not None:
            y = self.label_tranform(y)

        return x, y

    def get_data(self, name):
        """Utility for convenient data loading."""
        if name in ['train', 'unlabeled']:
            self.extract_images_labels(os.path.join(self.root, self.training_file))
        elif name == 'test':
            self.extract_images_labels(os.path.join(self.root, self.test_file))

    def extract_images_labels(self, filename):
        import gzip

        #print('Extracting', filename)
        with gzip.open(filename, 'rb') as f:
            raw_data = f.read().split()
        data = np.asarray([raw_data[start:start + 257]
                           for start in range(0, len(raw_data), 257)],
                          dtype=np.float32)
        images_vec = data[:, 1:]
        self.images = np.reshape(images_vec, (images_vec.shape[0], 16, 16))
        self.labels = data[:, 0].astype(int)
        self.images = ((self.images + 1)*128).astype('uint8')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.test_file))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.root, exist_ok=True)

        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            download_url(url, root=self.root,
                         filename=filename, md5=None)
        print('Done!')