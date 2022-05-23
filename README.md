# Deep Recurrent Neural Networks for Mortality Prediction in Intensive Care using Clinical Time Series at Multiple Resolutions

Tensorflow implementation to detect in-hospital mortality using EHR data as well as data from bedside monitor (ECG) using [MIMIC-III data](https://physionet.org/content/mimiciii/1.4/).

The unformatted code for the MTS-RNN model defined in our paper [Deep Recurrent Neural Networks for Mortality Prediction in Intensive Care using Clinical Time Series at Multiple Resolutions](https://drive.google.com/file/d/18VZa5ZhcinqK0sr4plwphBp1bjcKSHEP/view). The cleaned up version with instructions will be updated later. 

The work is based on the MGP-RNN model by Futoma et al. in their [paper](https://proceedings.mlr.press/v70/futoma17a/futoma17a.pdf) but used for task of ICU mortality prediction. 

The mgp-rnn implementation is using the code from their associated github repo. The data processing steps for using MIMIC-III dataset for ICU mortality prediction are based to benchmarking by [Harutyunan et al. 2019](https://github.com/YerevaNN/mimic3-benchmarks). 

After extracting episodes from MIMIC-III (similar to the benchmarking code), this repo also extracts the relevant waveforms from [MIMIC Waveform Matched SubSet](https://archive.physionet.org/physiobank/database/mimic3wdb/matched/). The files named as hierarchical_* perform the prediction using MTS-RNN. 


Citation:

```
@inproceedings{ghanvatkar2019deep,
  author={Ghanvatkar, Suparna and Rajan, Vaibhav},
  title={Deep Recurrent Neural Networks for Mortality Prediction in Intensive Care using Clinical Time Series at Multiple Resolutions},
  booktitle = {Proceedings of the 40th International Conference on Information Systems, ICIS 2019},
  publisher = {Association for Information Systems},
  year={2019},
  month={December 15-18},
  venue={Munich, Germany},
  isbn={978-0-9966831-9-7},
  url={https://aisel.aisnet.org/icis2019/data_science/data_science/12}
}
```
