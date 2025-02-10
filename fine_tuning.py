from image_transform import ImageTransform
from lib import *
from config import *
from utils import make_datapath_list, train_model, params_to_update
from dataset import *

def main():
    train_list = make_datapath_list("train")
    val_list = make_datapath_list("val")

    #dataset
    train_dataset = MyDataset(train_list, transform=ImageTransform(resize, mean, std), phase="train")
    val_dataset = MyDataset(val_list, transform=ImageTransform(resize, mean, std), phase="val")

    #dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

    dataloader_dict = {"train":train_dataloader, "val":val_dataloader}

    #network
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.classifier [6] = nn.Linear(in_features=4096, out_features=2, bias = True)

    #loss
    criterior = nn.CrossEntropyLoss ()


    #optimizer
    params1, params2, params3 = params_to_update(net)  
    # cập nhật tham số cho tầng features và lớp 1,4, và lớp cuối của tầng classifier
    optimizer = optim.SGD([
        {'params': params1, 'lr': 1e-4}, 
        {'params': params2, 'lr': 5e-4},
        {'params': params3, 'lr': 1e-3}, 
    ], momentum=0.9)
    
    train_model(net, dataloader_dict, criterior, optimizer, num_epochs)


if __name__ == "__main__":
    main()
    # network
    # use_pretrained = True
    # net = models.vgg16(pretrained=use_pretrained)
    # net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # load_model(net, save_path) tải trọng số đã lưu trước đó để dùng lại
    
    
    
    
    

# đây là đoạn code của transfer learning, chỉ cập nhật tham số cho lớp cuối


# params_to_update = []
# update_params_name = ["classifier.6.weight", "classifier.6.bias"]
# for name, param in net.named_parameters():
#     if name in update_params_name:
#         param.requires_grad = True
#         params_to_update.append(param)
#         print (name)
#     else:
#         param.requires_grad = False