# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor

from projects.sam_inference_demo import sam

from mmengine.runner.checkpoint import load_checkpoint
import torch
from mmengine import MessageHub
from ..utils import resize

model_zoo = {
    'base':
    'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-base-p16_3rdparty_sa1b-1024x1024_20230413-78a25eed.pth',  # noqa
    'large':
    'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-large-p16_3rdparty_sa1b-1024x1024_20230413-940520da.pth',  # noqa
    'huge':
    'https://download.openmmlab.com/mmsegmentation/v0.5/sam/sam_vit-huge-p16_3rdparty_sa1b-1024x1024_20230413-faaf96f6.pth',  # noqa
}


@MODELS.register_module()
class EncoderDecoderwithSAM_KD(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 SAM_arch: str = 'base',
                 SAM_config: OptConfigType = None,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 Prompt_backbone: ConfigType = None,
                 Prompt_neck: ConfigType = None,
                 Prompt_head: ConfigType = None,
                 Prompt_auxiliary_head: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        ## WARNING by LYU: DO NOT BUILD BACKBONE FROM CONFIG
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.SAM = self.init_SAM_model(SAM_config, SAM_arch)

        if Prompt_backbone is not None:
            self.Prompt_backbone = MODELS.build(Prompt_backbone)
        if Prompt_neck is not None:
            self.Prompt_neck = MODELS.build(Prompt_neck)
            self.with_Prompt_neck = True
        else:
            self.with_Prompt_neck = False
        if Prompt_head is not None:
            self.Prompt_head = MODELS.build(Prompt_head)
            self.with_Prompt_head = True
        else:
            self.with_Prompt_head = False
        if Prompt_auxiliary_head is not None:
            self.Prompt_auxiliary_head = MODELS.build(Prompt_auxiliary_head)
            self.with_Prompt_auxiliary_head = True
        else:
            self.with_Prompt_auxiliary_head = False
    
    def init_SAM_model(self, cfg: str, arch: str):
        model = MODELS.build(cfg)
        load_checkpoint(model, model_zoo.get(arch), strict=True)
        if torch.cuda.is_available():
            model = model.cuda()
        ## fix SAM encoder
        return model

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    ####################################
    ## Prompt student model train/val
    def prompt_extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.Prompt_backbone(inputs)
        if self.with_Prompt_neck:
            x = self.Prompt_neck(x)
        return x
    
    def _prompt_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        prompt_logits = self.Prompt_head(inputs)
        loss_decode = self.Prompt_head.loss_by_feat(prompt_logits, data_samples)
        #loss_decode = self.Prompt_head.loss(inputs, data_samples,
        #                                    self.train_cfg)

        losses.update(add_prefix(loss_decode, 'prompt_decode'))
        return prompt_logits, losses
    
    def _prompt_auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.Prompt_auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.Prompt_auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.Prompt_auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'prompt_aux'))

        return losses
    ####################################

    ####################################
    ## SAM teacher model train/val
    def extract_feat_SAM(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        image_embeddings = self.SAM.image_encoder(inputs)
        return image_embeddings
    
    def Prompt_encoder_forward(self, Prompt_segmentor_oup, size):
        outputs = dict()
        
        Prompt_segmentor_logits = resize(
            input=Prompt_segmentor_oup,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)

        Prompt_seg_mask = Prompt_segmentor_logits.argmax(dim=1) * 1.0
        Prompt_seg_mask = Prompt_seg_mask[:, None, :, :]
        ## sparse embeddings for the points and boxes, dense embeddings for the masks,
        sparse_embeddings, dense_embeddings = self.SAM.prompt_encoder(
                points=None,
                boxes=None,
                masks=Prompt_seg_mask,
            )
        outputs['se'] = sparse_embeddings
        outputs['de'] = dense_embeddings
        return outputs
    
    def Mask_decoder_forward(self, SAM_backbone_oup, Prompt_encoder_oup, size):
        outputs = dict()
        if isinstance(SAM_backbone_oup, tuple) or isinstance(SAM_backbone_oup, list):
            seg_fb = SAM_backbone_oup[-1]
        else:
            seg_fb = SAM_backbone_oup
        ## modified by LYU: aligne SAM feature and dense_pe_feature size
        if seg_fb.shape[2] != Prompt_encoder_oup['de'].shape[2] or seg_fb.shape[3] != Prompt_encoder_oup['de'].shape[3]:
            seg_embedding =  resize(
                input=seg_fb,
                size=Prompt_encoder_oup['de'].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        low_res_masks, iou_predictions = self.SAM.mask_decoder(
            image_embeddings=seg_embedding,
            image_pe=self.SAM.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=Prompt_encoder_oup['se'],
            dense_prompt_embeddings=Prompt_encoder_oup['de'],
            multimask_output=True,
        )
        masks = F.interpolate(
            low_res_masks, size, mode='bilinear', align_corners=False)
        ## just for Visualization
        self.Vis_Mask_decoder_out(masks)
        outputs['masks'] = masks
        outputs['low_res_masks'] = low_res_masks 
        return outputs
    
    def SAM_wPrompt_forward(self, SAM_backbone_fea, Prompt_m):
        outputs = dict()
        if self.with_neck:
            fea_neck = self.neck(SAM_backbone_fea)
        if fea_neck[0].shape[2] != Prompt_m['low_res_masks'].shape[2] or fea_neck[0].shape[3] != Prompt_m['low_res_masks'].shape[3]:
            low_res_prompt_mask =  resize(
                input=Prompt_m['low_res_masks'],
                size=fea_neck[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        pred = self.decode_head(fea_neck, low_res_prompt_mask)
        return pred
    
    def _SAM_head_forward_train(self, inputs_logits: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss_by_feat(inputs_logits, data_samples)
        losses.update(add_prefix(loss_decode, 'SAM_decode'))
        return losses
    
    @staticmethod
    def Vis_Mask_decoder_out(mask):
        import matplotlib.pyplot as plt
        #vis_mask = torch.max(mask, dim=1)[0]
        vis_mask = torch.mean(mask, dim=1)
        print(vis_mask.shape)
        #vis_mask = (vis_mask - vis_mask.mean())/(vis_mask.std())
        #vis_mask = self.sw_softmax(mask)
        vis_mask = vis_mask[0, :, :].detach().cpu().numpy()
        plt.figure(figsize=(15, 10))
        plt.subplot(221)
        plt.imshow(mask[0, 0, :, :].detach().cpu().numpy())
        plt.subplot(222)
        plt.imshow(mask[0, 1, :, :].detach().cpu().numpy())
        plt.subplot(223)
        plt.imshow(mask[0, 2, :, :].detach().cpu().numpy())
        plt.subplot(224)
        plt.imshow(vis_mask)
        plt.savefig("test.png")
    ####################################

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.prompt_extract_feat(inputs)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        
        #### Prompted only
        
        x = self.prompt_extract_feat(inputs)
        seg_logits = self.Prompt_head.predict(x, batch_img_metas,
                                              self.test_cfg)
        '''
        #### SAM with prompt
        x = self.prompt_extract_feat(inputs)
        prompt_logits = self.Prompt_head(x)
        SAM_backbone_out = self.extract_feat_SAM(inputs)
        PE_output_emb = self.Prompt_encoder_forward(prompt_logits, inputs.shape[2:])
        MD_output_mask = self.Mask_decoder_forward(SAM_backbone_out, PE_output_emb, inputs.shape[2:])
        SAM_segmentor_out = self.SAM_wPrompt_forward(SAM_backbone_out, MD_output_mask) 
        seg_logits = self.decode_head.predict_by_feat(SAM_segmentor_out, batch_img_metas)    
        '''
        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        message_hub = MessageHub.get_current_instance()
        curr_iter = message_hub.get_info('iter')
        
        losses = dict()
        ### 1. Prompt student forward and loss calculation
        x = self.prompt_extract_feat(inputs)
        prompt_logits, loss_decode = self._prompt_head_forward_train(x, data_samples)
        losses.update(loss_decode)
        if self.with_Prompt_auxiliary_head:
            loss_Prompt_aux = self._prompt_auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_Prompt_aux)
        ### 2. SAM forward and loss calculation
        if curr_iter > 100:
            #self.set_requires_grad(self.SAM, False)
            SAM_backbone_out = self.extract_feat_SAM(inputs)
            PE_output_emb = self.Prompt_encoder_forward(prompt_logits, inputs.shape[2:])
            MD_output_mask = self.Mask_decoder_forward(SAM_backbone_out, PE_output_emb, inputs.shape[2:])
            SAM_segmentor_out = self.SAM_wPrompt_forward(SAM_backbone_out, MD_output_mask) 
            loss_decode = self._SAM_head_forward_train(SAM_segmentor_out, data_samples)
            losses.update(loss_decode)
            ####################################
            ### KD on posterior probablities
            Prompt_segmentor_logits = resize(
                input=prompt_logits,
                size=SAM_segmentor_out.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ## 1. Teacher-Student
            TS_KL_loss = self.KL_loss(SAM_segmentor_out, Prompt_segmentor_logits, ratio=1, loss_name='TS-KL_loss')
            losses.update(TS_KL_loss)
            ## 2. Ensemble Teacher-Teacher
            eTeacher_logits = (SAM_segmentor_out + Prompt_segmentor_logits) / 2.0
            eTS_KL_loss = self.KL_loss(eTeacher_logits, SAM_segmentor_out, ratio=0.5, loss_name='eTS-KL_loss')
            losses.update(eTS_KL_loss)
            ####################################
            ####################################
            ### KD on intermediate feature maps
            SAM_backbone_fea = resize(
                input=SAM_backbone_out[3],
                size=x[3].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            SAM_backbone_fea_T = torch.mean(SAM_backbone_fea, dim=1)
            SAM_backbone_fea_T = SAM_backbone_fea_T[:, None, :, :]
            prompt_fea_S = torch.mean(x[3], dim=1)
            prompt_fea_S = prompt_fea_S[:, None, :, :]
            TS_MSE_loss = self.MSE_loss(SAM_backbone_fea_T, prompt_fea_S, ratio=0.5, loss_name='TS-MSE_loss')
            losses.update(TS_MSE_loss)
            ####################################

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        
        ## 1. prompted inference
        '''
        x = self.extract_feat(inputs)
        return self.Prompt_head.forward(x)
        '''
        '''
        ## 2. SAM inference
        SAM_backbone_out = self.extract_feat_SAM(inputs)
        if self.with_neck:
            fea_neck = self.neck(SAM_backbone_out)
            seg_logits = self.decode_head.forward(fea_neck)
            return seg_logits
        seg_logits = self.decode_head.forward(fea_neck)
        '''
        
        ## 3. SAM with prompt inference
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        x = self.prompt_extract_feat(inputs)
        prompt_logits = self.Prompt_head(x)
        SAM_backbone_out = self.extract_feat_SAM(inputs)
        PE_output_emb = self.Prompt_encoder_forward(prompt_logits, inputs.shape[2:])
        MD_output_mask = self.Mask_decoder_forward(SAM_backbone_out, PE_output_emb, inputs.shape[2:])
        SAM_segmentor_out = self.SAM_wPrompt_forward(SAM_backbone_out, MD_output_mask) 
        seg_logits = self.decode_head.predict_by_feat(SAM_segmentor_out, batch_img_metas)
        
        return seg_logits

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    ##########################################################
    ### Optimization techniques
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    ##########################################################

    ##########################################################
    ### Knowledge distillation techniques
    def MSE_loss(self, teacher, student, ratio=0.25, loss_name='MSE_loss'):
        losses = dict()
        L2_loss_dict = dict()
        MSE_loss = nn.MSELoss()
        #t = self.sw_softmax(teacher)
        #s = self.sw_softmax(student)
        t = teacher
        s = student
        L2_loss = MSE_loss(s, t)
        L2_loss_dict[loss_name] = ratio * L2_loss
        losses.update(L2_loss_dict)
        return losses
    
    def KL_loss(self, teacher, student, T=5, ratio=1.0, loss_name='KL_loss'):
        losses = dict()
        KL_loss_dict = dict()
        KL_loss = ratio * nn.KLDivLoss(reduction='mean')(F.log_softmax(student/T, dim=1),
                             F.softmax(teacher/T, dim=1)) * (T * T)
        KL_loss_dict[loss_name] = KL_loss
        losses.update(KL_loss_dict)
        return losses

    @staticmethod
    def sw_softmax(pred):
        N, C, H, W = pred.shape
        pred_sh = torch.reshape(pred, (N, C, H*W))
        pred_sh = F.softmax(pred_sh, dim=2)
        pred_out = torch.reshape(pred_sh, (N, C, H, W))
        return pred_out
    ##########################################################
