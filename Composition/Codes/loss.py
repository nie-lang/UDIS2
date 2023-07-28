import torch
import torch.nn as nn
import torch.nn.functional as F



# def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):

#     vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
#     if torch.cuda.is_available():
#         vgg_mean = vgg_mean.cuda()
#     vgg_input = input_255-vgg_mean
#     #x = vgg_model.features[0](vgg_input)
#     #FeatureMap_list.append(x)


#     for i in range(0,layer_index+1):
#         if i == 0:
#             x = vgg_model.features[0](vgg_input)
#         else:
#             x = vgg_model.features[i](x)

#     return x



def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def boundary_extraction(mask):

    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    #define kernel
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel,in_channel,3,3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    #dilation
    x = F.conv2d(1-mask,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)

    return x*mask

def cal_boundary_term(inpu1_tesnor, inpu2_tesnor, mask1_tesnor, mask2_tesnor, stitched_image):
    boundary_mask1 = mask1_tesnor * boundary_extraction(mask2_tesnor)
    boundary_mask2 = mask2_tesnor * boundary_extraction(mask1_tesnor)

    loss1 = l_num_loss(inpu1_tesnor*boundary_mask1, stitched_image*boundary_mask1, 1)
    loss2 = l_num_loss(inpu2_tesnor*boundary_mask2, stitched_image*boundary_mask2, 1)

    return loss1+loss2, boundary_mask1


def cal_smooth_term_stitch(stitched_image, learned_mask1):


    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(stitched_image[:,:,0:-1*delta,:] - stitched_image[:,:,delta:,:])
    dw_diff_img = torch.abs(stitched_image[:,:,:,0:-1*delta] - stitched_image[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss



def cal_smooth_term_diff(img1, img2, learned_mask1, overlap):

    diff_feature = torch.abs(img1-img2)**2 * overlap

    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(diff_feature[:,:,0:-1*delta,:] + diff_feature[:,:,delta:,:])
    dw_diff_img = torch.abs(diff_feature[:,:,:,0:-1*delta] + diff_feature[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss

    # dh_zeros = torch.zeros_like(dh_pixel)
    # dw_zeros = torch.zeros_like(dw_pixel)
    # if torch.cuda.is_available():
    #     dh_zeros = dh_zeros.cuda()
    #     dw_zeros = dw_zeros.cuda()


    # loss = l_num_loss(dh_pixel, dh_zeros, 1) + l_num_loss(dw_pixel, dw_zeros, 1)


    # return  loss, dh_pixel