import numpy as np

data = {
    (0.5, 5),
}

# Save the data to a file named "initial_params.npy"
np.save('initial_params.npy', data)



class ImageDataset(Dataset):
    def __init__(self, dir_path: str, transform: Optional[Callable] = None) -> None:
        super(ImageDataset, self).__init__()
        
        # Get a list of all files in the directory
        self.img_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, fname))]
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, 0)
        
        if img is None:
            raise Exception(f"Failed to load image at path: {img_path}")
        
        # Convert to PIL for compatibility with torchvision transforms
        img = Image.fromarray(img, mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img

    def __len__(self) -> int:
        return len(self.img_paths)


class ResNet34_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
            super(ResNet34_EmbeddingNet,self).__init__()
            modeling=no_grad(models.resnet34(pretrained=True))
            modules=list(modeling.children())[:-2]
            self.features=nn.Sequential(*modules)
            self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)
class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)
# Check if CUDA (GPU) is available
cuda = torch.cuda.is_available()

# If CUDA is available, move the model to the GPU
if cuda:
    device = torch.device("cuda:0")
    model=torch.load('model_windy.pth')
    model = model.to(device)

file_path = 'data_00'
data = '/'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # add any other transformations you might need
])

# Initialize dataset
dataset = ImageDataset(file_path+data, transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
]))

dataloader = DataLoader(dataset, shuffle=True)

def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        k = 0
        for images in dataloader:
            # If CUDA is available, move the images tensor to the GPU
            if cuda:
                images = images.to(device)
            embeddings[k:k+len(images)] = model.get_emdding(images).data.cpu().numpy()
            k += len(images)
    return embeddings

embeddings = extract_embeddings(dataloader, model)
print(type(embeddings))
print(embeddings)

# Save the numpy array to a file
np.save(os.path.join('embedding.npy'), embeddings)

import time
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, n_epochs=0, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    
    plt.scatter(embeddings[:,0], embeddings[:,1], alpha=0.5)
    
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    plt.show()


plot_embeddings(embeddings)
