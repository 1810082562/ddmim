from torchinfo import summary

import classifier
import ddmimv2
import models.ddmimv6
from sklearn.neighbors import KNeighborsClassifier
#model=classifier.Classifier(resume='DDMIM/log/seed1/version_44/checkpoints/last.ckpt')
# model=ddmimv2.DDMIMV2(
#         image_size=224,
#         patch_size=32,
#         tsfm_dim=1024,
#         stages_depth=[2,2,6,2],
#         heads=8,
#         mlp_dim=2048,
#         channels=3,
#         dim_head=128,
#         masking_ratio=0.5,
#         stagefactor=[1e-4,1e-3,0.01,0.1],
#         #lr=args.lr,
#         #momentum=args.momentum,
#         #weight_decay=args.weight_decay,
#         tau=0.5,
#         alpha=1
#         )

model

print(summary(models))
