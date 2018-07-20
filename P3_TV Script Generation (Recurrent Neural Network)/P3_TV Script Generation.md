
# P3 TV Script Generation

By Jen-Feng Hsieh

This project will generate the [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  The [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts have 27 seasons.  The Neural Network will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
import os
import pickle
import warnings

from collections import Counter
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
from tensorflow.contrib import seq2seq
```


```python
data_dir = './data/simpsons/moes_tavern_lines.txt'

input_file = os.path.join(data_dir)
with open(input_file, "r") as f:
    text = f.read()

# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data


```python
view_sentence_range = (0, 10)

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.248091603053435
    Number of lines: 4257
    Average number of words in each line: 11.50434578341555
    
    The sentences 0 to 10:
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    


## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, the first step is to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id: `vocab_to_int`
- Dictionary to go from the id to word: `int_to_vocab`


```python
def create_lookup_tables(text):
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse = True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab)}
    int_to_vocab = {ii: word for ii, word in enumerate(vocab)}
    return (vocab_to_int, int_to_vocab)
```

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  The dictionary will include the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word.


```python
def token_lookup():
    return {'.': '||Period||', 
            ',': '||Comma||', 
            '"': '||Quotation Mark||', 
            ';': '||Semicolon||', 
            '!': '||Exclamation mark||', 
            '?': '||Question mark||', 
            '(': '||Left Parentheses||', 
            ')': '||Right Parentheses||', 
            '--': '||Dash||', 
            '\n': '||Return||'}
```

## Preprocess all the data and save it


```python
token_dict = token_lookup()
for key, token in token_dict.items():
    text = text.replace(key, ' {} '.format(token))

text = text.lower()
text = text.split()
vocab_to_int, int_to_vocab = create_lookup_tables(text)
int_text = [vocab_to_int[word] for word in text]

pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))
```

# Check Point


```python
int_text, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
```

## Build the Neural Network
The components which are necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    Default GPU Device: /gpu:0


### Input
Create TF Placeholders for the Neural Network.
- Input text placeholder
- Targets placeholder
- Learning Rate placeholder


```python
def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')    
    return (inputs, targets, learning_rate)
```

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)


```python
def get_init_cell(batch_size, rnn_size, keep_prob = 0.8, lstm_layers = 1):
    lstm_size = rnn_size
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    return (cell, initial_state)
```

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed
```

### Build RNN
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)


```python
def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    return (outputs, final_state)
```

### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    embed = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn = None)
    return (logits, final_state)
```

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches is a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

The last batch will be dropped if there is not enough data for the last batch.

For exmple, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

The last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    n_batches = len(int_text) // (batch_size * seq_length)
    int_text = int_text[: n_batches * batch_size * seq_length]
    int_text.append(int_text[0])
    batches = []
    for ii in range(n_batches):
        inputs = []
        targets = []
        for jj in range(batch_size):
            start = (ii + jj * n_batches) * seq_length
            inputs.append(int_text[start : start + seq_length])
            targets.append(int_text[start + 1 : start + seq_length + 1])
        batches.append([inputs, targets])
    return np.array(batches)
```

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
num_epochs = 100
batch_size = 128
rnn_size = 256
embed_dim = 256
seq_length = 32
learning_rate = 0.01

# Show stats for every n number of batches
show_every_n_batches = 128

save_dir = './save'
```

### Build the Graph


```python
train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.


```python
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/18   train_loss = 8.822
    Epoch   7 Batch    2/18   train_loss = 3.039
    Epoch  14 Batch    4/18   train_loss = 2.141
    Epoch  21 Batch    6/18   train_loss = 1.605
    Epoch  28 Batch    8/18   train_loss = 1.301
    Epoch  35 Batch   10/18   train_loss = 1.062
    Epoch  42 Batch   12/18   train_loss = 0.956
    Epoch  49 Batch   14/18   train_loss = 0.782
    Epoch  56 Batch   16/18   train_loss = 0.694
    Epoch  64 Batch    0/18   train_loss = 0.604
    Epoch  71 Batch    2/18   train_loss = 0.590
    Epoch  78 Batch    4/18   train_loss = 0.570
    Epoch  85 Batch    6/18   train_loss = 0.584
    Epoch  92 Batch    8/18   train_loss = 0.534
    Epoch  99 Batch   10/18   train_loss = 0.592
    Model Trained and Saved


## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
pickle.dump((seq_length, save_dir), open('params.p', 'wb'))
```

# Checkpoint


```python
_, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
seq_length, load_dir = pickle.load(open('params.p', mode='rb'))
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    inputs = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return (inputs, initial_state, final_state, probs)
```

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    return int_to_vocab[np.argmax(probabilities)]
```

## Generate TV Script


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    moe_szyslak:(clears throat) it is, though: one. i have been happier...
    chief_wiggum:(reading)" the troll twins of underbridge academy."!
    carl_carlson:(nods) moe, the can meet all of the party why where you gave me?
    homer_simpson:(singing) when this happened, now?
    barney_gumble: i got it! we're proud! two of us that told you.
    carl_carlson: i could forget that fish snout.
    moe_szyslak:(to moe) hey, handsome, send the bill to my most excited, and low-blow boxing for the window?
    moe_szyslak:(pointed) let me check that some pian-ee.
    moe_szyslak: homer, you just experienced w. r. o. l. first... edna asked me to talk to you.
    seymour_skinner:(warmly) eh, screw it time.
    homer_simpson:(grateful) thanks, moe.
    moe_szyslak: hey,


The result shows the content of predicted TV script is nonsensical because we only trained on fewer data. In order to get good results, I will train models in further steps using more data that have less vocabulary.
