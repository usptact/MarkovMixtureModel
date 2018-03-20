# MarkovMixtureModel
A sample Markov Mixture Model implementation in Infer.NET (C#)

The model accepts a number of sequences with discrete states and clusters them.

Each sequence in a cluster is modeled using initial state probabilities and a transition probability matrix. Globally, the model also learns the cluster proportion.

## Usage

1. Prepare a data file or use the synthetic data generator
2. Run the modeling binary

## Data Format
The data file is a newline-delimited text file where each line represents a sequence. The states of a sequence are integers that are space-delimited.

```
<sequence>
<sequence> ::= <state> [<state> .. [..]]
<state> ::= {0,1,...}
```

(Work in progress)
