# Debiasing


## Cosine Similarity

Given two input words it determines how similar those two entities are. I have chosen gender based exampled, so the first step was to find the vector difference between man and woman and store them in a variable,g .

`g = word_to_vec_map['woman'] - word_to_vec_map['man']`

The input list of names given- 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin','mario','steve','sekar'

Their cosine similarities with genders,

```
ronaldo -0.3124479685032943
priya 0.17632041839009402
rahul -0.16915471039231722
danielle 0.24393299216283892
reza -0.07930429672199552
katy 0.2831068659572615
yasmin 0.23313857767928758
mario -0.21908983683176902
steve -0.3589033919583839
sekar 0.007634727568022135
```

Negative values shows the names are similar to woman and positive values indicate the names are closer to man.



