from rfmid_dataset import RFMiDDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = RFMiDDataset(
    csv_file='Retinal-disease-classification/labels.csv',
    img_dir='Retinal-disease-classification/images',
    transform=transform
)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)