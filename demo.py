import torch
from torchvision.transforms import *
from model import Net
from torchvision.utils import save_image
    
def load_img(filepath):
    img = Image.open(filepath)
    return img

def main():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    model = torch.load('checkpoint/model_epoch_36.pth')['model'].to(device)
    img = load_img('test.jpg')
    transform = transforms.Compose([transforms.ToTensor()])
    model_in = transform(img).unsqueeze(0).to(device)
    model_out = model(model_in).cpu()
    
    save_image(model_out, 'out.jpg')
    
if __name__=='__main__':
    main()