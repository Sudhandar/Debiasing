import numpy as np

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
            
    return words, word_to_vec_map

words, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v

    """

    dot = np.dot(u,v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    cosine_similarity = dot/(norm_u*norm_v)
    
    return cosine_similarity

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task : a is to b as c is to ____. 
    
    """
    
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a, e_b, e_c = word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -10
    best_word = None

    input_words_set = set([word_a, word_b, word_c])
    
    for w in words:        
        if w in input_words_set:
            continue
        
        cosine_sim = cosine_similarity(e_b-e_a,word_to_vec_map[w]-e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
            
    return best_word

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
    
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g)

print ('List of names and their similarities with constructed vector:')

name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
    
print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
    
def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.
    
    """

    e = word_to_vec_map[word]
    e_biascomponent = (np.dot(e,g)/np.sum(g**2))*g
    e_debiased = e-e_biascomponent
    
    return e_debiased

e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    """

    w1, w2 = pair[0],pair[1]
    e_w1, e_w2 = word_to_vec_map[w1],word_to_vec_map[w2]

    mu = (e_w1+e_w2)/2

    mu_B = (np.dot(mu,bias_axis)/np.sum(bias_axis**2))*bias_axis
    mu_orth = mu-mu_B

    e_w1B =  (np.dot(e_w1,bias_axis)/np.sum(bias_axis**2))*bias_axis
    e_w2B =  (np.dot(e_w2,bias_axis)/np.sum(bias_axis**2))*bias_axis
        
    corrected_e_w1B = np.sqrt(np.abs(1-np.sum(mu_orth**2)))*((e_w1B-mu_B)/np.linalg.norm((e_w1-mu_orth)-mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1-np.sum(mu_orth**2)))*((e_w2B-mu_B)/np.linalg.norm((e_w2-mu_orth)-mu_B))

    e1 = corrected_e_w1B+mu_orth
    e2 = corrected_e_w2B+mu_orth
                                                                    
    return e1, e2

print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))