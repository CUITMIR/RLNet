import os
from PIL import Image
import io

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset


class Sichuan128(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.is_train = is_train

        this_dir = os.path.dirname(os.path.realpath(__file__))

        self.data_root = os.path.join(this_dir, '../../data/sc/images')

        if is_train:
            with open(os.path.join(this_dir, '../../data/sc/train_set.txt'), 'r') as f:
                self.dataset = eval(f.read())
        else:
            with open(os.path.join(this_dir, '../../data/sc/val_set.txt'), 'r') as f:
                self.dataset = eval(f.read())

        self.mean = 0
        self.std = 1
        self.data_name = 'sc_128'

    def __getitem__(self, idx):
        filenames = self.dataset[idx]

        x = []
        for i in range(10):
            path = os.path.join(self.data_root, filenames[i] + '.pt')
            x.append(torch.load(path).float())  # C H W
        x = torch.stack(x, dim=0)  # T C H W

        y = []
        for i in range(10, 20):
            path = os.path.join(self.data_root, filenames[i] + '.pt')
            y.append(torch.load(path).float())  # C H W
        y = torch.stack(y, dim=0)  # T C H W

        return x, y

    def __len__(self):
        return len(self.dataset)


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 128, 128],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    train_set = Sichuan128(is_train=True)
    valid_set = Sichuan128(is_train=False)

    dataloader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    dataloader_vali = torch.utils.data.DataLoader(
        valid_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False
    )
    dataloader_test = torch.utils.data.DataLoader(
        valid_set,
        batch_size=val_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=False
    )

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataloader_train, _, dataloader_test = load_data(batch_size=4, val_batch_size=4, num_workers=4)

    print(len(dataloader_train), len(dataloader_test))
    print(len(dataloader_train.dataset), len(dataloader_test.dataset))

    # cnt = 0
    # for item in dataloader_train:
    #     print(item[0].shape, item[1].shape)

        # fig, ax = plt.subplots(4, 5, figsize=(5 * 3, 4 * 3))
        # for i in range(2):  # x
        #     for j in range(5):
        #         ax[i][j].imshow(item[0][0, i * 5 + j, ...].permute(1, 2, 0), cmap='jet')
        # for i in range(2):  # y
        #     for j in range(5):
        #         ax[i + 2][j].imshow(item[1][0, i * 5 + j, ...].permute(1, 2, 0), cmap='jet')
        # plt.show()
        #
        # for row in item[0][0, 0, 0, 64:96, 64:96]:
        #     print(row)
        #
        # cnt += 1
        # if cnt > 0:
        #     break
    # for item in dataloader_test:
    #     print(item[0].shape, item[1].shape)
    #
    #     fig, ax = plt.subplots(4, 5, figsize=(5 * 3, 4 * 3))
    #     for i in range(2):  # x
    #         for j in range(5):
    #             ax[i][j].imshow(item[0][0, i * 5 + j, ...].permute(1, 2, 0), cmap='jet')
    #     for i in range(2):  # y
    #         for j in range(5):
    #             ax[i + 2][j].imshow(item[1][0, i * 5 + j, ...].permute(1, 2, 0), cmap='jet')
    #     plt.show()
    #
    #     cnt += 1
    #     if cnt > 0:
    #         break
