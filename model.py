from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F

# simplest thing should be to predict a segmentation mask first
class SegmentationBaseline(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, vision_encoder, text_encoder):
        super(SegmentationBaseline, self).__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        self.mm_fusion = None
        self.mm_decoder = None

    # position - curr vehcle pos 3d -> (proj trans) 2d curr pos
    # next vehicle pos 3d -> (proj trans) 2d next pos
    # offset - 2d offset
    # in eval - 2d to 3d inverse proj trans
    def forward(self, frames, text):
        
        vision_feat = self.vision_encoder(frames)
        text_feat = self.text_encoder(text)
        
        fused_feat = self.mm_fusion(vision_feat, text_feat)
        segm_mask = self.mm_decoder(fused_feat)

        return segm_mask
    

# TODO
class FullBaseline(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, vision_encoder, text_encoder):
        super(FullBaseline, self).__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        self.mm_fusion = None
        self.mm_decoder = None
        
        self.destination_extractor = None
        self.position_net = None

    # position - curr vehcle pos 3d -> (proj trans) 2d curr pos
    # next vehicle pos 3d -> (proj trans) 2d next pos
    # offset - 2d offset
    # in eval - 2d to 3d inverse proj trans
    def forward(self, frames, text, position):
        
        vision_feat = self.vision_encoder(frames)
        text_feat = self.text_encoder(text)
        
        fused_feat = self.mm_fusion(vision_feat, text_feat)
        segm_mask = self.mm_decoder(fused_feat)
        
        destination = self.destination_extractor(segm_mask)
        position_offset = self.position_net(position, destination)

        return segm_mask, destination, position_offset