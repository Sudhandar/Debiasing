# Debiasing


## Cosine Similarity

Given two input words it determines how similar those two entities are. I have chosen gender based example, so the first step was to find the vector difference between man and woman and store them in a variable,g .

`g = word_to_vec_map['woman'] - word_to_vec_map['man']`

A list of names were given as input - 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin','mario','steve','sekar'

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

A list of general English words were given as input - 'lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist',  'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer'

Their cosine similarities with genders,

```
lipstick 0.2769191625638266
guns -0.1888485567898898
science -0.060829065409296994
arts 0.008189312385880328
literature 0.06472504433459927
warrior -0.20920164641125288
doctor 0.11895289410935041
tree -0.07089399175478091
receptionist 0.33077941750593737
technology -0.13193732447554296
fashion 0.03563894625772699
teacher 0.17920923431825664
engineer -0.08039280494524072
pilot 0.0010764498991916787
computer -0.10330358873850498
singer 0.1850051813649629
```

## Word analogy

Performs sentence completion as follows , *a is to b as c is to _*

Input : 'italy', 'italian', 'spain'

Output : ```italy -> italian :: spain -> spanish```

Input : ''small', 'smaller', 'large'

Output : ```small -> smaller :: large -> larger```

## Neutralize

Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. Lets take receptionist as an example,

```
cosine similarity between receptionist and g, before neutralizing:  0.33077941750593737

cosine similarity between receptionist and g, after neutralizing:  -2.920516166121757e-17 (close to zero)
```

## Debiasing (equalize)

Debias gender specific words.

```
cosine similarities before equalizing:
cosine_similarity(word_to_vec_map["man"], gender) =  -0.1171109576533683
cosine_similarity(word_to_vec_map["woman"], gender) =  0.3566661884627037

cosine similarities after equalizing:
cosine_similarity(e1, gender) =  -0.7004364289309386
cosine_similarity(e2, gender) =  0.7004364289309387

```

Source for the Glove vector : (https://nlp.stanford.edu/projects/glove/)
