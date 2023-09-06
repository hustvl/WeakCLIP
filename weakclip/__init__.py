from .weakclip import WeakCLIP
from .models import CLIPResNet, CLIPTextEncoder, CLIPVisionTransformer, CLIPResNetWithAttention
from .heads import IdentityHead, FPNHeadWithFeatures, FPNHeadDGCN, SegHeadDGCN, IdentityHeadDGCN
from .fpn import FPNDGCNBaseline
from .dgcn import DGCNBaseline
from .my_encoder_decoder import EncoderDecoderInferenceNpy
from .coco import COCODataset
from .res2net import Res2Net
# import dgcnutils