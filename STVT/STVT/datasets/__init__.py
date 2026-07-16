from .TVSum import TVSum
from .SumMe import SumMe
from .SumMe_i3d_FR import SumMe_i3d_FR
from .SumMe_Rgb_Flow_Resnet import SumMe_Rgb_Flow_Resnet
from .SumMe_normalized_concat_rfr import SumMe_normalized_concat_rfr
from .SumMe_Rgb_Flow_Resnet_matched import  SumMe_Rgb_Flow_Resnet_matched
from .SumMe_RFR_Normalized import SumMe_RFR_Normalized  #with normalized featuresPCA contributions:
#   ResNet:   31.5% 
#   I3D RGB:  41.9% 
#   I3D Flow: 26.5%  

from .TVSum_i3d_FR import TVSum_i3d_FR
from .TVSum_Rgb_Flow_Resnet import TVSum_Rgb_Flow_Resnet
from .TVSum_Rgb_Flow import TVSum_Rgb_Flow


from .TVSum_RFR_Normalized import TVSum_RFR_Normalized


__all__ = ['TVSum','SumMe','SumMe_i3d_FR','TVSum_i3d_FR',
           'TVSum_Rgb_Flow_Resnet','SumMe_Rgb_Flow_Resnet','TVSum_Rgb_Flow','SumMe_Rgb_Flow_Resnet_matched','SumMe_normalized_concat_rfr',
           'SumMe_RFR_Normalized','TVSum_RFR_Normalized']

