import tqdm
from torch import optim
from net import *
from torchvision.utils import save_image
from dataset import load_data_voc, class_nums, VOC_COLORMAP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=device)
    X = pred.long()
    return colormap[X, :]


weight_path = 'params/unet.pth'
data_path = r"E:\myDataset\VOCdevkit\VOC2012"
save_path = r'E:\U-Net\train_image'
if __name__ == '__main__':
    train_data, test_data = load_data_voc(voc_dir=data_path, batch_size=10, crop_size=(256, 256))
    net = UNet(n_channels=3, n_classes=class_nums).to(device)

    opt = optim.Adam(net.parameters())
    loss_fun = nn.CrossEntropyLoss()

    epoch = 1
    while epoch < 200:
        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image, segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i % 100 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')
                _image = image[0]
                _segment_image = label2image(segment_image[0]).permute(2, 0, 1)
                _out_image = label2image(out_image[0].argmax(dim=0)).permute(2, 0, 1)
                aaa = out_image[0].argmax(dim=0)
                img = torch.stack([_image, _segment_image, _out_image], dim=0)
                save_image(img, f'{save_path}/{epoch}-{i}.png')

        if epoch % 50 == 0:
            torch.save(net.state_dict(), weight_path)
            print('save successfully!')
        epoch += 1
