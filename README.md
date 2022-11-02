# Speech_Inversion_TCN

This repository holds the baseline models used in the "THE SECRET SOURCE : INCORPORATING SOURCE FEATURES TO IMPROVE
ACOUSTIC-TO-ARTICULATORY SPEECH INVERSION"  paper. Current baseline models are from the ones trained on the XRMB articulatory dataset. The same model archtectures were used with the HPRC dataset for the experiments in the paper.

This repository is still under construction !!

## Baseline models

[BiGRNN-MFCC](model_BGRU.py) : Model trained with MFCCs as inputs and 6 TVs as target outputs

[BiGRNN-SF-MFCC](model_BGRU_ext.py) : Model trained with MFCCs as inputs and 6 TVs + source features as target outputs

[CNN-BiGRNN-Mspec](model_CNN-BiGRNN.py) : Model trained with melspectrograms as inputs and 6 TVs as target outputs

[CNN-BiGRNN-SF-Mspec](model_CNN-BiGRNN_ext.py) : Model trained with melspectrograms as inputs and 6 TVs + source features as target outputs

[CNN-BLSTM-Mspec](model_CNN-BLSTM.py) : Model trained with melspectrograms as inputs and 6 TVs as target outputs

[CNN-BLSTM-SF-Mspec](model_CNN-BLSTM_ext.py) : Model trained with melspectrograms as inputs and 6 TVs + source features as target outputs