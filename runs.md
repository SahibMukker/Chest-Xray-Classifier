Jan 1st 2026 Run 1
- best VAL AUROC/mAP/micro-f1: macro_AUROC=0.9578, macro_mAP=0.8887
- TEST AUROC/mAP/micro-f1: macro_AUROC=0.7264, macro_mAP=0.2409
- Fixed thr=0.50 | VAL micro-F1=0.9129 | TEST micro-F1=0.3698
  Fixed thr=0.35 | VAL micro-F1=0.9104 | TEST micro-F1=0.3771
  Fixed thr=0.25 | VAL micro-F1=0.9087 | TEST micro-F1=0.3828 
- checkpoint filename: best_valF1-0.919_testF1-0.352_before_nanfix.pt
- Best VAL global threshold: 0.95 | val_micro_f1=0.9152
  test_loss: 10.9742 | test_micro_f1@0.95: 0.3026

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-01-13 — Run 2 (NaN Patient ID fix, 224, BCE baseline)
**Change:** removed NaN Patient IDs before splitting; verified image overlap(train_val_list vs test_list)=0  
**Input size:** 224×224  
**Model/backbone:** (fill in)  
**Pretrained:** ImageNet  
**Loss:** BCEWithLogitsLoss + pos_weight  
**Augmentation:** none (no horizontal flip)  
**Epochs trained:** 50  
**Best checkpoint:** epoch=22 | best_val_micro_f1@0.5=0.522545344006485  
**Resume note:** evaluation loaded checkpoint from `runs/best_ever.pt`

**VAL:** macro AUROC=0.7454 | macro mAP=0.2258  
**TEST:** macro AUROC=0.7291 | macro mAP=0.2461  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.5170 | TEST micro-F1=0.3968
- Fixed thr=0.50 → VAL micro-F1=0.5224 | TEST micro-F1=0.3720
- Fixed thr=0.35 → VAL micro-F1=0.5259 | TEST micro-F1=0.3838
- Fixed thr=0.25 → VAL micro-F1=0.5269 | TEST micro-F1=0.3902
- Best VAL global threshold=0.25 → val_micro_f1=0.5269 | test_micro_f1=0.3902

**TEST:** loss=8.9945 | micro-F1@0.25=0.3902

**Notes**
- VAL and TEST AUROC are close (no big overfitting), overall performance limited by low mAP.
- Best test micro-F1 achieved around threshold ~0.25–0.35.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-01-14 — Run 3 (Focal + 384 + light aug, batch size reduced)
**Change vs prior baseline:** swapped BCE → FocalLossWithLogits; resize 224→384; added light train aug; reduced batch size due to VRAM
**Input size:** 384×384
**Transforms**
- Train: Resize(384,384) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(384,384) + Normalize(ImageNet)
**Loss:** FocalLossWithLogits (gamma=2.0, alpha=None) + pos_weight
**Batch size:** (16)
**Epochs trained:** 50
**Best checkpoint:** epoch=48 | best_val_micro_f1@0.5=0.5097610791811144

**VAL:** macro AUROC=0.7776 | macro mAP=0.2347
**TEST:** macro AUROC=0.7470 | macro mAP=0.2524

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.5011 | TEST micro-F1=0.4015
- Fixed thr=0.50 → VAL micro-F1=0.5097 | TEST micro-F1=0.3856
- Fixed thr=0.35 → VAL micro-F1=0.5112 | TEST micro-F1=0.4088  (best TEST)
- Fixed thr=0.25 → VAL micro-F1=0.4903 | TEST micro-F1=0.4044
- Best VAL global threshold=0.40 → val_micro_f1=0.5153 | test_micro_f1@0.40=0.4015

**Notes**
- Clear improvement vs BCE+224 baseline: TEST micro-F1 best ~0.409 at thr=0.35.
- Best checkpoint very late (epoch 48/50) → consider LR scheduling before extending epochs.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-01-22 — Run 4 (Focal + 384 + crop + aug + accum2)
**Change vs prior focal384_aug:** added RandomResizedCrop(384, scale=(0.85,1.0)); enabled gradient accumulation (accum_steps=2)
**Input size:** 384×384
**Transforms**
- Train: RandomResizedCrop(384, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(384,384) + Normalize(ImageNet)
**Loss:** FocalLossWithLogits (gamma=2.0, alpha=None) + pos_weight
**Batch size:** 16
**Accum steps:** 2  (effective batch = 32)
**Epochs trained:** 50
**Best checkpoint:** epoch=34 | best_val_micro_f1@0.5=0.5147111250917479

**VAL:** macro AUROC=0.7869 | macro mAP=0.2474  
**TEST:** macro AUROC=0.7621 | macro mAP=0.2655  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.5188 | TEST micro-F1=0.4151  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=0.5147 | TEST micro-F1=0.4080
- Fixed thr=0.35 → VAL micro-F1=0.4767 | TEST micro-F1=0.4026
- Fixed thr=0.25 → VAL micro-F1=0.4280 | TEST micro-F1=0.3771
- Best VAL global threshold=0.50 → val_micro_f1=0.5147 | test_micro_f1@0.50=0.4080

**TEST:** loss=2.1359 | micro-F1@0.50=0.4080

**Notes**
- Best jump is in TEST macro mAP (0.2655), indicating improved ranking.
- Micro-F1 roughly stable; best TEST achieved with per-class thresholds tuned on val.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-01-23 — Run 5 (gamma 1.5)
**Change vs prior focal384_aug:** changed gamma( 2.0 -> 1.5)
**Input size:** 384×384
**Transforms**
- Train: RandomResizedCrop(384, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(384,384) + Normalize(ImageNet)
**Loss:** FocalLossWithLogits (gamma=1.5, alpha=None) + pos_weight
**Batch size:** 16
**Accum steps:** 2  (effective batch = 32)
**Epochs trained:** 50
**Best checkpoint:** epoch _____

**VAL:** macro AUROC=_____ | macro mAP=_____
**TEST:** macro AUROC=_____ | macro mAP=_____  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=_____ | TEST micro-F1=_____  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=_____ | TEST micro-F1=_____
- Fixed thr=0.35 → VAL micro-F1=_____ | TEST micro-F1=_____
- Fixed thr=0.25 → VAL micro-F1=_____ | TEST micro-F1=_____
- Best VAL global threshold=_____ → val_micro_f1=0._____ | test_micro_f1@0.50=_____

**TEST:** loss=_____ | micro-F1@0.50=_____

**Notes**

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-02-11 — Run 6 (512x512)
**Change vs prior focal384_aug:** changed resolution from 384x384 -> 512x512
**Input size:** 512x512
**Transforms**
- Train: RandomResizedCrop(512, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(512,512) + Normalize(ImageNet)
**Loss:** FocalLossWithLogits (gamma=1.5, alpha=None) + pos_weight
**Batch size:** 8
**Accum steps:** 4  (effective batch = 32)
**Epochs trained:** 50
**Best checkpoint:** epoch=48 | best_val_micro_f1@0.5=0.5019446970723118

**VAL:** macro_AUROC=0.7959 | macro_mAP=0.2529
**TEST:** macro_AUROC=0.7670 | macro_mAP=0.2759  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.5123 | TEST micro-F1=0.4058  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=0.5020 | TEST micro-F1=0.4043
- Fixed thr=0.35 → VAL micro-F1=0.4475 | TEST micro-F1=0.3846
- Fixed thr=0.25 → VAL micro-F1=0.3926 | TEST micro-F1=0.3554
- Best VAL global threshold=0.50  → val_micro_f1=0.5020

**TEST:** test_loss: 1.8426 | test_micro_f1@0.50: 0.4043

**Notes** 
- 512x512 resolution improved AUROC/mAP ratings (0.2655 -> 0.2759) but not micro f1 (~0.40-0.41). 
- Next: reduce resolution to 384x384, focus on top 4-5 prevalant diseases (diseases with most data) to see if micro f1 improves, and then look to increase label count, stronger backbone, etc.

----------------------------------------------------------------------------------------------------------------------------------------------------------------

feb 26th: 224x224, top 5 diseases

## 2026-02-26 — Run 7 (224x224)
**Change vs prior focal384_aug:** resolution 512x512 -> 224x224, training on top 5 diseases
**Input size:** 224x224
**Transforms**
- Train: RandomResizedCrop(224, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(224,224) + Normalize(ImageNet)
**Loss:** FocalLossWithLogits (gamma=1.5, alpha=None) + pos_weight
**Batch size:** 8
**Accum steps:** 4  (effective batch = 32)
**Epochs trained:** 50
**Best checkpoint:** epoch=13 | best_val_micro_f1@0.5=0.5879145114078351

**VAL:** macro_AUROC=0.7753 | macro_mAP=0.4503
**TEST:** macro_AUROC=0.7800 | macro_mAP=0.4589  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.6013 | TEST micro-F1=0.6096  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=0.5879 | TEST micro-F1=0.5949
- Fixed thr=0.35 → VAL micro-F1=0.5508 | TEST micro-F1=0.5550
- Fixed thr=0.25 → VAL micro-F1=0.5072 | TEST micro-F1=0.5108
- Best VAL global threshold=0.50  → val_micro_f1=0.5879

**TEST:** test_loss: 0.4486 | test_micro_f1@0.50: 0.5949

**Notes** 
- Val macro AUROC went down slightly but every other metric improved drastically. Best epoch was early on though so more training wont improve. Likely that it is converging too fast and thats why best epoch was so early in training.
- Next: increase resolution to 384x384?

----------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-03-2 — Run 8 (384x384)
**Change vs prior focal384_aug:** resolution 224x224 -> 384x384, training on top 5 diseases
**Input size:** 224x224
**Transforms**
- Train: RandomResizedCrop(384, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(384,384) + Normalize(ImageNet)
**Loss:** FocalLossWithLogits (gamma=1.5, alpha=None) + pos_weight
**Batch size:** 8
**Accum steps:** 4  (effective batch = 32)
**Epochs trained:** 25
**Best checkpoint:** epoch=11 | best_val_micro_f1@0.5=0.588156862745098

**VAL:** macro_AUROC=0.7941 | macro_mAP=0.4794
**TEST:** macro_AUROC=0.7962 | macro_mAP=0.4816  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.6013 | TEST micro-F1=0.6096  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=0.5882 | TEST micro-F1=0.5932
- Fixed thr=0.35 → VAL micro-F1=0.5275 | TEST micro-F1=0.5271
- Fixed thr=0.25 → VAL micro-F1=0.4686 | TEST micro-F1=0.4691
- Best VAL global threshold=0.50  → val_micro_f1=0.5884

**TEST:** test_loss: 0.3415 | test_micro_f1@0.55: 0.5887

**Notes** 
- increasing to 384x384 improved AUROC and mAP, but micro-f1 dropped slightly. model still converging too fast.
- Next steps: stick with 224x224 and change focal loss to BCEwithLogits to see if focal loss is even doing anything.
  - if focal loss is better than BCE, mess around with gamma values (1.0, 1.5, 2.0)

---------------------------------------------------------------------------------------------------------------------------------------------------------------
-
## 2026-03-2 — Run 9 (224x224)
**Change vs prior focal384_aug:** resolution 224x224, training on top 5 diseases, focal loss to BCE
**Input size:** 224x224
**Transforms**
- Train: RandomResizedCrop(224, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(224, 224) + Normalize(ImageNet)
**Loss:** BCEWithLogits + pos_weight
**Batch size:** 32
**Accum steps:** 1  (effective batch = 32)
**Epochs trained:** 25
**Best checkpoint:** epoch=13 | best_val_micro_f1@0.5=0.6053282987203692

**VAL:** macro_AUROC=0.7843 | macro_mAP=0.4629
**TEST:** macro_AUROC=0.7862 | macro_mAP=0.4640  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.6106 | TEST micro-F1=0.6112  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=0.6053 | TEST micro-F1=0.6067
- Fixed thr=0.35 → VAL micro-F1=0.5985 | TEST micro-F1=0.5992
- Fixed thr=0.25 → VAL micro-F1=0.5869 | TEST micro-F1=0.5871
- Best VAL global threshold=0.50  → val_micro_f1=0.6053

**TEST:** test_loss: 1.3419 | test_micro_f1@0.50: 0.6067

**Notes** 
- AUROC and mAP dropped slightly but big improvements in micro-f1.
- Next steps: Run test one more time to make sure its not a fluke (p < 0.05) and then ablations maybe?
- Realized that I have been at 384x384 still and not 224x224 so need to redo once again

---------------------------------------------------------------------------------------------------------------------------------------------------------------

## 2026-04-10 — Run 9 (224x224)
**Change vs prior focal384_aug:** resolution 224x224, training on top 5 diseases, focal loss to BCE
**Input size:** 224x224
**Transforms**
- Train: RandomResizedCrop(224, scale=(0.85,1.0)) + RandomRotation(5) + ColorJitter(brightness=0.1, contrast=0.1) + Normalize(ImageNet)
- Val/Test: Resize(224, 224) + Normalize(ImageNet)
**Loss:** BCEWithLogits + pos_weight
**Batch size:** 32
**Accum steps:** 1  (effective batch = 32)
**Epochs trained:** 25
**Best checkpoint:** epoch=3 | best_val_micro_f1@0.5=0.5915727362130268

**VAL:** macro_AUROC=0.7959 | macro_mAP=0.4844
**TEST:** macro_AUROC=0.7953 | macro_mAP=0.4864  

**Thresholding / micro-F1**
- Per-class tuned thresholds → VAL micro-F1=0.6143 | TEST micro-F1=0.6230  (best TEST)
- Fixed thr=0.50 → VAL micro-F1=0.5916 | TEST micro-F1=0.6009
- Fixed thr=0.35 → VAL micro-F1=0.5541 | TEST micro-F1=0.5601
- Fixed thr=0.25 → VAL micro-F1=0.5226 | TEST micro-F1=0.5234
- Best VAL global threshold=0.60  → val_micro_f1=0.5999

**TEST:** test_loss: 0.8524 | test_micro_f1@0.60: 0.6117

**Notes** 
- since best epoch was 3/25, next train session do training from scratch with smaller lr