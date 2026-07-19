# My SumMe datasets:
from .SumMe import SumMe
from .SumMe_RFR_Normalized import SumMe_RFR_Normalized  #with normalized featuresPCA contributions:
from .SumMe_Resnet_Flow import SumMe_Resnet_Flow
from .SumMe_Resnet_Rgb import SumMe_Resnet_Rgb
# end
from .SumMe_Rgb_Flow_Resnet import SumMe_Rgb_Flow_Resnet
from .SumMe_Rgb_Flow_Resnet_matched import  SumMe_Rgb_Flow_Resnet_matched

#   ResNet:   31.5% 
#   I3D RGB:  41.9% 
#   I3D Flow: 26.5%  

# My TVSum datasets:
from .TVSum import TVSum
from .TVSum_Resnet_Rgb import TVSum_Resnet_Rgb
from .TVSum_Resnet_Flow import TVSum_Resnet_Flow
from .TVSum_RFR_Normalized import TVSum_RFR_Normalized
#end

from .TVSum_Rgb_Flow_Resnet import TVSum_Rgb_Flow_Resnet




__all__ = ['TVSum','SumMe','SumMe_Resnet_Flow',
           'TVSum_Rgb_Flow_Resnet','SumMe_Rgb_Flow_Resnet',
           'SumMe_Rgb_Flow_Resnet_matched','SumMe_Resnet_Rgb',
           'SumMe_RFR_Normalized',
           'TVSum_RFR_Normalized','TVSum_Resnet_Flow''TVSum_Resnet_Rgb',]

