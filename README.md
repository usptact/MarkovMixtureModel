The model can be used to cluster discrete value sequences into a pre-defined number of clusters. The cluster assignments are soft.

Each sequence in a cluster is modeled using (a) a vector of initial state probabilities and (b) a transition probability matrix. Globally, the model also learns (c) the proportion of the clusters.

The assumption is that a sequence belongs only to one cluster. The parameters generating a sequence don't change within a sequence (no switching).

## Usage

1. Prepare a data file or use the synthetic data generator
2. Run the modeling binary

`MarkovMixtureModel.exe -data:data.txt -clusters:5 -predictions:pred.txt`

3. Sequence cluster assignments can be found in `pred.txt`

## Data Format
The data file is a newline-delimited text file where each line represents a sequence. The states of a sequence are integers that are space-delimited.

```
<sequence>
<sequence> ::= <state> [<state> .. [..]]
<state> ::= {0,1,...}
```

Updated to use open source Infer.NET
