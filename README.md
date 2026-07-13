
## Usage

First, download the dataset file  

Prepared .h5 file   

Please put them into "./STVT/datasets/datasets"  
## Instructions for Code:

```
cd STVT
train.py --roundtimes save_name --dataset TVSum or SumMe
```
The eval is included in training.py   

## Reference 

Please cite the following papers when you apply the code. 

[1] T.-C. Hsu, Y.-S. Liao and C.-R. Huang, "Video Summarization With Spatiotemporal Vision Transformer," IEEE Transactions on Image Processing, vol. 32, pp. 3013-3026, 2023, doi: 10.1109/TIP.2023.3275069.

[2] T.-C. Hsu, Y.-S. Liao and C.-R. Huang, "Video Summarization With Frame Index Vision Transformer," in Proc. International Conference on Machine Vision and Applications (MVA), Aichi, Japan, 2021, pp. 1-5, 2021, doi: 10.23919/MVA51890.2021.9511350.



python train.py --dataset SumMe --epochs 2 --batch_size 4
