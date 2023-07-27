import argparse
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, Network
from dataset import TrainDataset
import glob
from loss import cal_lp_loss, inter_grid_loss, intra_grid_loss



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
    
    # define dataset
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
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            inpu1_tesnor = batch_value[0].float()
            inpu2_tesnor = batch_value[1].float()

            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()

            # forward, backward, update weights
            optimizer.zero_grad()

            batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
            # result
            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']

            # calculate loss for overlapping regions
            overlap_loss = cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            # calculate loss for non-overlapping regions
            nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)

            total_loss = overlap_loss + nonoverlap_loss
            total_loss.backward()

            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            loss_sigma += total_loss.item()

            print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / score_print_fre
                average_overlap_loss = overlap_loss_sigma/ score_print_fre
                average_nonoverlap_loss = nonoverlap_loss_sigma/ score_print_fre
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)

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

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default='/opt/data/private/nl/Data/UDIS-D/training/')

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)


