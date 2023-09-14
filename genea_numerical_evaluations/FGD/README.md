## Fréchet Gesture Distance (FGD)

Scripts to calculate FGD for the GENEA Gesture Generation Challenge 2020 submissions.
We follow the FGD implementation in [Speech Gesture Generation from the Trimodal Context of Text, Audio, and Speaker Identity (ACM TOG, 2020)](https://arxiv.org/abs/2009.02119).

### Environment
* Ubuntu 18.04, Python 3.6, Pytorch 1.7.1

### Run
1. Prepare data. Put Trinity Gesture Dataset on `data/Trinity`, which will be used to train an autoencoder. Put challenge system results on `data/Cond_??`. 
2. Train an autoencoder. You can set `n_frames` in `train_AE.py` to change the number of frames in a sample. 
   ```bash
   $ python train_AE.py
   ```
3. Calculate FGD.
   ```bash
   $ python evaluate_FGD.py
   ```

### Results
The FGD values for three different `n_frames=30,60,90`. We also report FGD on raw data space similar to the one used in [No Gestures Left Behind: Learning Relationships between Spoken Language and Freeform Gestures (EMNLP Findings 2020)](https://www.aclweb.org/anthology/2020.findings-emnlp.170.pdf). Lower FGD is better.
```text
----- EXP (n_frames: 30) -----
FGD on feature space and raw data space
Cond_BA: 12.989, 343.584
Cond_BT: 35.315, 395.650
Cond_SA: 21.402, 175.214
Cond_SB: 19.480, 167.150
Cond_SC:  9.970, 119.424
Cond_SD:  4.778, 106.149
Cond_SE: 11.086, 112.048

----- EXP (n_frames: 60) -----
FGD on feature space and raw data space
Cond_BA: 12.986, 718.180
Cond_BT: 30.301, 817.556
Cond_SA: 21.654, 393.294
Cond_SB: 19.016, 374.749
Cond_SC:  9.410, 277.185
Cond_SD:  4.120, 246.324
Cond_SE: 10.854, 256.386

----- EXP (n_frames: 90) -----
FGD on feature space and raw data space
Cond_BA: 11.777, 1126.280
Cond_BT: 27.311, 1272.092
Cond_SA: 18.155, 658.897
Cond_SB: 17.559, 633.881
Cond_SC:  8.641, 477.692
Cond_SD:  3.620, 428.553
Cond_SE:  9.952, 442.145
```
