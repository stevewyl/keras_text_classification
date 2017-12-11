import numpy as np
import re


# clean useless characters
'''
html_clean = ['& ldquo ;', '& hellip ;', '& rdquo ;', '& yen ;']
punctuation_replace = '[，。！？]+'
strange_num = ['①','②','③','④']
'''
punctuation_remove = '[：；……（）『』《》【】～!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'

def clean(sent):
    sent = re.sub(r'ldquo', "", sent)
    sent = re.sub(r'hellip', "", sent)
    sent = re.sub(r'rdquo', "", sent)
    sent = re.sub(r'yen', "", sent)
    sent = re.sub(r'⑦', "7", sent)
    sent = re.sub(r'(， ){2,}', "", sent)
    sent = re.sub(r'(！ ){2,}', "", sent) # delete too many！，？，。等
    sent = re.sub(r'(？ ){2,}', "", sent)
    sent = re.sub(r'(。 ){2,}', "", sent)
    sent = re.sub(punctuation_remove, "", sent) #delete punctuations
    s = ' '.join(sent.split()) #delete additional space
    
    return s
    
def sent_filter(l):
    l_new = []
    for s,k in enumerate(l):
        if len(k) > 2:
            l_new.append(k)
    return l_new
        
def load_data_and_labels(good_data_file, bad_data_file, mid_data_file):
    #load reviews and save them in the list
    good_examples = list(open(good_data_file, "r", encoding='utf-8').readlines())
    good_examples = [s.strip() for s in good_examples]
    bad_examples = list(open(bad_data_file, "r", encoding='utf-8').readlines())
    bad_examples = [s.strip() for s in bad_examples]
    mid_examples = list(open(mid_data_file, "r", encoding='utf-8').readlines())
    mid_examples = [s.strip() for s in mid_examples]

    #Call the clean () and sent_filter () functions to process the comments, save them in the x_text list
    good_examples = [clean(sent) for sent in good_examples]
    bad_examples = [clean(sent) for sent in bad_examples]
    mid_examples = [clean(sent) for sent in mid_examples]

    good_examples = [i.strip() for i in good_examples]
    bad_examples = [i.strip() for i in bad_examples]
    mid_examples = [i.strip() for i in mid_examples]

    good_examples = sent_filter(good_examples)
    bad_examples = sent_filter(bad_examples)
    mid_examples = sent_filter(mid_examples)

    x_text = good_examples + bad_examples + mid_examples

    #Add a label for each comment and save it in y
    good_labels = [[1, 0, 0] for _ in good_examples]
    bad_labels = [[0, 1, 0] for _ in bad_examples]
    mid_labels = [[0, 0, 1] for _ in mid_examples]
    y = np.concatenate([good_labels, bad_labels, mid_labels], 0)
    return [x_text, y]

# when you use tensorflow, you need to generate batches yourself, this function may helpe you
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
