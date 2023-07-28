import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
import glob
from loss import cal_boundary_term, cal_smooth_term_stitch, cal_smooth_term_diff



# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))

# path to save the summary files
SUMMARY_DIR = os.path.join(last_path, 'summary')
writer = SummaryWriter(log_dir=SUMMARY_DIR)

# path to save the model files
MODEL_DIR = os.path.join(last_path, 'model')

# create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)



def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    train_data = TrainDataset(data_path=args.train_path)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)

    # define the network
    net = Network()

    if torch.cuda.is_available():
        net = net.cuda()


    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')



    print("##################start training#######################")
    score_print_fre = 300

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        sigma_total_loss = 0.
        sigma_boundary_loss = 0.
        sigma_smooth1_loss = 0.
        sigma_smooth2_loss = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            warp1_tensor = batch_value[0].float()
            warp2_tensor = batch_value[1].float()
            mask1_tensor = batch_value[2].float()
            mask2_tensor = batch_value[3].float()

            if torch.cuda.is_available():
                warp1_tensor = warp1_tensor.cuda()
                warp2_tensor = warp2_tensor.cuda()
                mask1_tensor = mask1_tensor.cuda()
                mask2_tensor = mask2_tensor.cuda()


            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net,  warp1_tensor,  warp2_tensor, mask1_tensor, mask2_tensor)

            learned_mask1 = batch_out['learned_mask1']
            learned_mask2 = batch_out['learned_mask2']
            stitched_image = batch_out['stitched_image']

            # boundary term
            boundary_loss, boundary_mask1 = cal_boundary_term( warp1_tensor,  warp2_tensor, mask1_tensor, mask2_tensor, stitched_image)
            boundary_loss = 10000 * boundary_loss

            #  smooth term
            # on stitched image
            smooth1_loss = cal_smooth_term_stitch(stitched_image, learned_mask1)
            smooth1_loss = 1000* smooth1_loss
            # on different image
            smooth2_loss = cal_smooth_term_diff( warp1_tensor,  warp2_tensor, learned_mask1, mask1_tensor*mask2_tensor)
            smooth2_loss = 1000 * smooth2_loss


            total_loss = boundary_loss + smooth1_loss + smooth2_loss
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()


            sigma_boundary_loss += boundary_loss.item()
            sigma_smooth1_loss += smooth1_loss.item()
            sigma_smooth2_loss += smooth2_loss.item()
            sigma_total_loss += total_loss.item()

            print(glob_iter)
            # print loss etc.
            if i % score_print_fre == 0 and i != 0:
                average_total_loss = sigma_total_loss / score_print_fre
                average_boundary_loss = sigma_boundary_loss/ score_print_fre
                average_smooth1_loss = sigma_smooth1_loss/ score_print_fre
                average_smooth2_loss = sigma_smooth2_loss/ score_print_fre

                sigma_total_loss = 0.
                sigma_boundary_loss = 0.
                sigma_smooth1_loss = 0.
                sigma_smooth2_loss = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}   boundary loss: {:.4f}  smooth loss: {:.4f}  diff loss: {:.4f}   lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader), average_total_loss, average_boundary_loss, average_smooth1_loss, average_smooth2_loss, optimizer.state_dict()['param_groups'][0]['lr']))

                # visualization
                writer.add_image("inpu1", (warp1_tensor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (warp2_tensor[0]+1.)/2., glob_iter)


                writer.add_image("stitched_image", (stitched_image[0]+1.)/2., glob_iter)
                writer.add_image("learned_mask1", learned_mask1[0], glob_iter)
                writer.add_image("boundary_mask1", boundary_mask1[0], glob_iter)


                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_total_loss, glob_iter)
                writer.add_scalar('average_boundary_loss', average_boundary_loss, glob_iter)
                writer.add_scalar('average_smooth1_loss', average_smooth1_loss, glob_iter)
                writer.add_scalar('average_smooth2_loss', average_smooth2_loss, glob_iter)

            glob_iter += 1


        scheduler.step()
        # save model
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    print("##################end training#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()

    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/training')

    #nl: parse the arguments
    args = parser.parse_args()
    print(args)

    print('<==================== jump into training function ===================>\n')
    #nl: rain
    train(args)


