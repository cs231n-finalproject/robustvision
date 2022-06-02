"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
from sklearn.metrics import SCORERS
import torch

from vis_misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class CamExtractorTransConv():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None

        B = x.shape[0]

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.model.maxpool(self.model.act1(self.model.bn1(self.model.conv1(x))))

        # 1 stage
        cls_tokens = self.model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.model.has_pre_trained_vit:
            x_t = self.model.trans_patch_conv(x)
            x_t = torch.cat((cls_tokens, x_t), dim=1)
            x_t = x_t + self.model.pos_embed
            x_t = self.model.pos_drop(x_t)
        else:
            x_t = self.model.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
            x_t = torch.cat([cls_tokens, x_t], dim=1)
        x = self.model.conv_1(x_base, return_x_2=False)
        x_t = self.model.trans_1(x_t)

        # 2 ~ final 
        for i in range(2, self.model.fin_stage):         
            x, x_t = eval('self.model.conv_trans_' + str(i))(x, x_t)
            if int(i) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer            

        # final ~ vit_final
        for i in range(self.model.fin_stage, self.model.vit_final_stage):         
            x_t = self.model.pre_trained_vit.blocks[i-1](x_t)

        return conv_output, x, x_t


        # for module_pos, module in self.model.features._modules.items():
        #     x = module(x)  # Forward
        #     if int(module_pos) == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer
        # return conv_output, x


    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x, x_t = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # # Forward pass on the classifier
        # x = self.model.classifier(x)
        # return conv_output, x        

        # conv classification
        x_p = self.model.pooling(x).flatten(1)
        conv_cls = self.model.conv_cls_head(x_p)

        # trans classification
        if self.model.has_pre_trained_vit:        
            if self.model.global_pool:
                x_t = x_t[:, 1:, :].mean(dim=1)  # global pool without cls token
                x_t = self.model.trans_norm(x_t)
            else:
                x_t = self.model.trans_norm(x_t)
                x_t = x_t[:, 0]
        else:
            x_t = self.model.trans_norm(x_t)
            x_t = x_t[:, 0]
        tran_cls = self.model.trans_cls_head(x_t)

        scores = conv_cls + tran_cls   

        return conv_output, scores


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        # self.extractor = CamExtractor(self.model, target_layer)
        self.extractor = CamExtractorTransConv(self.model, target_layer)        

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()     
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 1  # cat dog
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example, model='alexnet')
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example, model='transconv')        
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=12)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
