# MarkovMixtureModel
A sample Markov Mixture Model implementation in Infer.NET (C#)

The model can be used to cluster discrete value sequences into a pre-defined number of clusters. The cluster assignments are soft.

Each sequence in a cluster is modeled using (a) a vector of initial state probabilities and (b) a transition probability matrix. Globally, the model also learns the proportion of the clusters.

## Usage

1. Prepare a data file or use the synthetic data generator
2. Run the modeling binary

### Training

`MarkovMixtureModel.exe -mode:train -data:data.txt -model:model.dat -predictions:predictions.txt`

The model `data.txt` and cluster assignments `predictions.txt` are written after the model is trained.

### Prediction

`MarkovMixtureModel.exe -mode:predict -model:model.dat -data:new_data.dat -predictions:new_predictions.txt`

Given the model `model.dat` and a data file `new_data.dat`, the cluster assignments are written to `new_predictions.txt`.

Note: The number of discrete states must the same as used during the training!

## Data Format
The data file is a newline-delimited text file where each line represents a sequence. The states of a sequence are integers that are space-delimited.

```
<sequence>
<sequence> ::= <state> [<state> .. [..]]
<state> ::= {0,1,...}
```

(Work in progress)
