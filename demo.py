import torch
from torchvision.transforms import *
from model import Net
from torchvision.utils import save_image
from PIL import Image
import numpy as np
    
def resize_img(img_np, height, width, resample=Image.BICUBIC):
    img_np = np.asarray(img_np)
    img = Image.fromarray(img_np)
    resized = img.resize((width, height), resample=resample)
    return np.array(resized)


def load_img(filepath):
    img = Image.open(filepath)
    return img


def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = torch.load('checkpoint/model_epoch_36.pth')['model'].to(device)
    img = load_img('test.jpg')
    w, h = img.size[0], img.size[1]
    img = resize_img(img, h*2, w*2)
    print(h*2,w*2)
    transform = transforms.Compose([transforms.ToTensor()])
    model_in = transform(img).unsqueeze(0).to(device)
    model_out = model(model_in).cpu()
    
    save_image(model_out, 'out.jpg')
    
if __name__=='__main__':
    main()