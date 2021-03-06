## Code associated to the paper "On Neural Architectures for Astronomical Time-series Classification with Application to Variable Stars"
By Jamal & Bloom (2020)


This code is associated to the publication Jamal & Bloom (2020) on neural network architectures for variables stars classification.
The code provides architectures for _direct classifiers_, _autoencoders_ and _composite networks_ with variants of layer types: RNN (Recurrent Neural Networks), tCNN (temporal Convolutional Neural Networks) and dTCN (dilated Temporal Convolutional Networks).

The paper focuses on applications of _direct classifiers_ and _composite networks_ trained on fixed-length data (batch-mode).
In our design, _autoencoders_ and _composite networks_ require fixed-size inputs to comply with implementation specifications of the decoder module. 
The network identified as _direct classifiers_ can process fixed-size data in a batch mode, as well as a list of observables with different lengths using generator functions. 
The generator functions are provided in this code.

Notebooks, directed for users, are made available to (1) download the data from the public MACHO VS database, (2) run examples of preprocessing functions and (3) train networks.
The latter task requires to load the preprocessed data of the MACHO variable stars database used in the paper (17603 objects in total) that is made available in a Zenodo repository (doi:10.5281/zenodo.3908783). 


For batch computation on terminal, shell scripts are provided in progs/.

```
cd deepnets_vs/
```

-  _direct classifiers_ trained on the B-band data (phase-folded) through generator functions:
```
./progs/list_foldedLCs/run_MACHO_ClassifierNet_tCNN.sh           blue tCNN 32 3
./progs/list_foldedLCs/run_MACHO_ClassifierNet_RNN.sh            blue LSTM 32 1
./progs/list_foldedLCs/run_MACHO_ClassifierNet_RNN.sh            blue GRU  16 2
```

- networks trained on reduced B-band data (normalized, phase-folded, fixed-length):
```
./progs/matrix_norm_foldedLCs/run_MACHO_ClassifierNet_tCNN.sh    blue tCNN 32 3
./progs/matrix_norm_foldedLCs/run_MACHO_ClassifierNet_RNN.sh     blue LSTM 32 1
./progs/matrix_norm_foldedLCs/run_MACHO_ClassifierNet_RNN.sh     blue GRU  16 2

./progs/matrix_norm_foldedLCs/run_MACHO_CompositeNet_tCNN.sh     blue tCNN 32 3
./progs/matrix_norm_foldedLCs/run_MACHO_CompositeNet_RNN.sh      blue LSTM 32 1
./progs/matrix_norm_foldedLCs/run_MACHO_CompositeNet_RNN.sh      blue GRU  16 2
```

- networks trained on reduced B-band data (raw, phase-folded, fixed-length):
```
./progs/matrix_raw_foldedLCs/run_MACHO_ClassifierNet_tCNN.sh    blue tCNN 32 3
./progs/matrix_raw_foldedLCs/run_MACHO_ClassifierNet_RNN.sh     blue LSTM 32 1
./progs/matrix_raw_foldedLCs/run_MACHO_ClassifierNet_RNN.sh     blue GRU  16 2

./progs/matrix_raw_foldedLCs/run_MACHO_CompositeNet_tCNN.sh     blue tCNN 32 3
./progs/matrix_raw_foldedLCs/run_MACHO_CompositeNet_RNN.sh      blue LSTM 32 1
./progs/matrix_raw_foldedLCs/run_MACHO_CompositeNet_RNN.sh      blue GRU  16 2
```

Arguments in command line refer to:
- the type of data (B-band:'blue', merged R- and B-band: 'rb', a hybrid variant with R- and B-band: 'multiple')
- the model type (LSTM, GRU, tCNN and dTCN)
- the network size (integer) 
- the number of layers/stacks (integer)

Shell files in /progs can be adapted to train networks on a wider range of hyperparameters.


__NOTES__

Sections of this code refer to previous work from Naul, Bloom et al (2017, 2018). 
Headers of python src files cite proper credits. 


__REFERENCES__

- MACHO public database: http://macho.nci.org.au/
- Alcock, C., Allsman, R. A., Axelrod, T. S., et al. 1996, ApJ, 461, 84, doi:10.1086/177039
- Jamal, S. & Bloom, J. S., 2020, arXiv:2003.08618
- Naul, B., Bloom, J. S., Pérez, F., & van der Walt, S. 2017, Zenodo, doi:10.5281/zenodo.1045560
- Naul, B., Bloom, J. S., Pérez, F., & van der Walt, S. 2018, Nat. Astron., 2, 151

Exhaustive list of references and acknowledgments are cited in Jamal & Bloom (2020).


__ACKNOWLEDGMENTS on the use of the public MACHO VS data__

_This work utilizes public domain data obtained by the MACHO Project, jointly funded by the US Department of Energy through the University of California, Lawrence Livermore National Laboratory under contract No. W-7405-Eng-48, by the National Science Foundation through the Center for Particle Astrophysics of the University of California under cooperative agreement AST-8809616, and by the Mount Stromlo and Siding Spring Observatory, part of the Australian National University._



