# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:46:18 2018

@author: Carlos
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import re
import html
import string
from tensorflow.contrib import keras
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
import pickle
import json

from DataGenerator import DataGenerator


class FacebookLSTM:
    
    def __init__(self):
        
        # For consistent file names
        self.prefix = ''
        
        # Mapping data
        self.char_indices = {}
        self.indices_char = {}
        self.vocab_size = 0
        
        # Sequence data
        self.seq_len = 40
        self.integer_sequences = None
        self.integer_labels = None
        
        # LSTM Model
        self.model = None
        
        
###############################################################################
#                Create, Continue, or Load an LSTM model
###############################################################################
        
    def create_model(filename, prefix = None,
                     seq_len = 40, max_chars = None):
        '''
        Creates and trains an LSTM model from a Facebook message JSON export
        
        Input:
            @filename: Name of the JSON facebook export file
            @prefix: How to save created model and mappings
            @seq_len: Sequence size used to predict next character
            @max_chars: To reduce size of training data for faster training
        Output:
            @return: Created FacebookLSTM object
        '''
        
        # Create new instance
        faceb_lstm = FacebookLSTM()
        faceb_lstm.seq_len = seq_len
        
        # Generate prefix if we had none
        faceb_lstm.__generate_prefix(prefix, filename)
        
        # Parse JSON into formatted text
        df = FacebookLSTM.load_json(filename)
        text = faceb_lstm.__format_text(df, max_chars)
        
        # Generate mappings
        faceb_lstm.__generate_mappings(text)
        
        # Create and encode sequences
        sequences, next_chars = FacebookLSTM.generate_sequences(
                text = text, length = seq_len)
        
        faceb_lstm.__encode_sequences(sequences, next_chars)
        
        # Create and train LSTM
        faceb_lstm.__create_lstm()
        faceb_lstm.__train_lstm()
        
        # Return the created instance
        return(faceb_lstm)
        
    
    def continue_model(filename, prefix = None, 
                          seq_len = 40, max_chars = None):
        '''
        Continues training an existing LSTM model
        
        Input:
            @filename: Name of the JSON facebook export file
            @prefix: How to save created model and mappings
            @seq_len: Sequence size used to predict next character
            @max_chars: To reduce size of training data for faster training
        Output:
            @return: Created FacebookLSTM object
        '''
        
        # Create new instance
        faceb_lstm = FacebookLSTM()
        faceb_lstm.seq_len = seq_len
        
        # Determine prefix, then load saved model and mapping files
        faceb_lstm.__generate_prefix(prefix, filename)
        faceb_lstm.__load_model_mappings()
        
        # Parse JSON into formatted text
        df = FacebookLSTM.load_json(filename)
        text = faceb_lstm.__format_text(df, max_chars)
        
        # Create and encode sequences
        sequences, next_chars = FacebookLSTM.generate_sequences(
                text = text, length = seq_len)
        
        faceb_lstm.__encode_sequences(sequences, next_chars)
        
        # Continue training model
        faceb_lstm.__train_lstm()
        
        # Return the created instance
        return(faceb_lstm)
        
    
    def load_model(filename, prefix = None, seq_len = 40, max_chars = None):
        '''
        Loads an existing LSTM model to generate text with.
        
        Input:
            @filename: Name of the JSON facebook export file
            @prefix: How to save created model and mappings
            @seq_len: Sequence size used to predict next character
            @max_chars: To reduce size of training data for faster training
        Output:
            @return: Created FacebookLSTM object
        '''
        
        # Create new instance
        faceb_lstm = FacebookLSTM()
        faceb_lstm.seq_len = seq_len
        
        # Determine prefix, then load saved model and mapping files
        faceb_lstm.__generate_prefix(prefix, filename)
        faceb_lstm.__load_model_mappings()
        
        # Parse JSON into formatted text
        df = FacebookLSTM.load_json(filename)
        text = faceb_lstm.__format_text(df, max_chars)
        
        # Create and encode sequences
        sequences, next_chars = FacebookLSTM.generate_sequences(
                text = text, length = seq_len)
        
        faceb_lstm.__encode_sequences(sequences, next_chars)
        
        # Return prepared instance
        return(faceb_lstm)
        
        
###############################################################################
#                   Internal Workings of the System
###############################################################################
        
    def __generate_prefix(self, prefix, filename):
        '''
        Uses either the given prefix, or uses the file name (without extension)
        The prefix is used for saving the model and mappings with a consistent
        naming convension.
        
        Input:
            @prefix: Option to supply a prefix
            @filename: Name of JSON file being loaded
        Output:
            @post: Consistent prefix stored in object
        '''
        
        print('Generating Prefix...')
        
        if prefix is None:
            self.prefix = re.findall('(.*).json',
                                     filename)[0]
        else:
            self.prefix = prefix
            
        print()
        
        
    def load_json(filepath):
        '''
        Converts a Facebook messages JSON output file to a pandas dataframe
        
        Input:
            @filepath: File name of the Facebook JSON output file
        Output:
            @return: Facebook messages as pandas dataframe
        '''
        
        print('Loading JSON...')
        
        # Load messages from json
        with open(filepath) as f:
            file = json.load(f)
        
        messages = file['messages']
        
        # Convert to dataframe
        df = pd.DataFrame.from_dict(messages)
        
        print()
        
        return(df)

    
    def __format_text(self, df, max_chars = None):
        '''
        Transforms facebook messages dataframe to a string of all messages
        
        Input:
            @df: Facebook messages dataframe
            @max_chars: Option to limit text string length for memory saving
        Output:
            @return: Chronological messages as a single string
        '''
    
        print('Formating text...')
        
        # Prepare regex to strip all weird characters out
        p = ['\\' + x for x in string.punctuation]
        punct = ''.join(p)
        regex_str = '[^a-z0-9 ' + punct + ']'
        pattern = re.compile(regex_str)
        
        # Edit each row
        messages = []
        for index, row in df.iterrows():
            
            # Skip rows without text content
            if pd.isnull(row.content):
                continue
            
            # Edit content
            message = html.unescape(row.content)
            message = message.lower()
            message = re.sub(pattern, '', message)
            messages.append(row.sender_name + ": " + message)
    
        # Oldest to Newest
        messages.reverse()
        
        # Convert to text
        text = '\n'.join(messages)
        print("Text size is: ", len(text))
        print()
        
        # Limit number of characters used (if desired)
        if max_chars is not None:
            text = text[-max_chars:]
        
        return(text)
        
    
    def __generate_mappings(self, text):
        '''
        Creates mappings from integers to characters and vice versa. Also
        determines the vocabulary size, and saves mappings as a pickle.
        
        Input:
            @text: Entire facebook message log as a formatted string
        Output:
            @post: 
                - {character -> indices} dictionary saved to class and disk
                - {indices -> character} dictionary saved to class and disk
                - Vocabulary size saved to class
        '''
        
        print('Generating mappings...')
        
        # Prepare dictionaries to map chars to indicies, and vice versa
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char_indices = {c: i for i, c in enumerate(chars)}
        self.indices_char = {i: c for i, c in enumerate(chars)}
        
        print('Vocabulary Size: %d' % self.vocab_size)
        print()
        
        # Saving mappings for future use
        ci_name = '_'.join([self.prefix, 'char_indices.pkl'])
        ic_name = '_'.join([self.prefix, 'indices_char.pkl'])
        pickle.dump(self.char_indices, open(ci_name, 'wb'))
        pickle.dump(self.indices_char, open(ic_name, 'wb'))


    def generate_sequences(text, length = 40, step = 1):
        '''
        Convert the giant wall of text into semi redundant sequences of characters
        with their corresponding output. eg.
        
           Sequences              next_chars
        -------------------------------------
        Matthew Hennenga    ->        n
        thew Hennegan: H    ->        e
        w Hennegan : Hel    ->        l
        
        Inputs:
            @text: Text to convert into sequences
            @length: Size of window, how many letters to use to predict
            @step: Increase to reduce training time
        Output:
            @return:
                - sequences: List of sequences generated from text
                - next_chars: List of character which followed the sequence
        '''
        
        print("Generating sequences from text...")
        
        # Parameters for the sequence generation
        sequences = []      # List of sequences to use as input
        next_chars = []     # List of characters that sequences predicted
    
        # Split text into sequences, with a list of outputs from that sequence
        for i in range(0, len(text) - length, step):
            sequences.append(text[i: i + length])
            next_chars.append(text[i + length])
            
        print('Number of Sequences:', len(sequences))
        print()
    
        return(sequences, next_chars)


    def __encode_sequences(self, sequences, next_chars):
        '''
        Convert text sequences to their integer encodings
        
        Input:
            @sequences: A list of all text sequences to be encoded
            @next_chars: A list of all following characters to be encoded
        Output:
            @post:
                - All sequences encoded as integers and saved to class
                - All sequences labels encoded as integers and saved to class 
        '''
        
        print("Encoding sequences and labels as integers...")
        
        # Map each character to its integer representation
        encoded_sequences = [[self.char_indices[char] for char in seq] for seq in sequences]
        encoded_labels    = [self.char_indices[char] for char in next_chars]
    
        # Store as numpy arrays
        self.integer_sequences = np.array(encoded_sequences)
        self.integer_labels    = np.array(encoded_labels)
        
        print()


    def __create_lstm(self):
        '''
        Initialize and compile new LSTM network ready to be trained
        
        Output:
            @post: LSTM model initialized and saved to class
        '''
        
        print("Creating LSTM neural network...")
        
        # Define model
        model = keras.models.Sequential()
        in_shape = (self.seq_len, self.vocab_size)
        model.add(keras.layers.LSTM(128, input_shape = in_shape))
        model.add(keras.layers.Dense(self.vocab_size, activation='softmax'))
        print(model.summary())
        
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        # Save model
        self.model = model
        
    
    def __train_lstm(self):
        '''
        Train compiled LSTM model with class data.
        
        Uses a DataGenerator to work with large quantities of data which 
        could cause memory issues.
        
        After each training epoch, the model has sample text printed for 
        manual inspection using a variety of diversity values.
        
        It also saves the model to file after each epoch allowing the
        user to stop training at any time, and seamlessly continue training
        later using the continue_model method.
        '''
        
        print("Training LSTM model...")
        
        # Name to save model as
        model_name = '_'.join([self.prefix, 'model.h5'])
        
        # Create generator for model
        data_gen = DataGenerator(self.integer_sequences,
                                 self.integer_labels,
                                 self.vocab_size)
        
        # Train incrementally prining out sample output
        num_epochs = 100
        for i in range(num_epochs):
            print("Training Epoch: ", i)
            
            # Train next epoch and save
            self.model.fit_generator(generator=data_gen,
                                     epochs = 1,
                                     verbose = 2)
            self.model.save(model_name)
            
            # Test how well the model is currently doing
            seed = data_gen.get_seed_text()
            self.__test_model(seed)
            
    
    
    def __test_model(self, seed):
        '''
        Print out text using the current model for manual inspection.
        
        Uses a variety of diversity values, as the best may differ on the
        amount of training data used.
        
        Input:
            @seed: The seed int encoded sequence to use to begin text generation
        Output:
            @post: Generated text printed to screen
        '''
        
        # Test each diversity to see what performs best
        test_diversities = [0.2, 0.5, 1, 1.2]
        for diversity in test_diversities:
            print("-" * 40)
            print("Testing model with diversity:", diversity)
            print()
            
            text = self.__generate_seq(seed, n_chars = 250,
                                     diversity = diversity)
            
            print(text)
            print("-" * 40)
            
    
    def __generate_seq(self, seed_int_encode, n_chars = 250, diversity = 0.2):
        '''
        Generate text using the current LSTM model
        
        Input:
            @seed_int_encode: Seed sequence encoded as integers
            @n_chars: Number of characters to generate
            @diversity: How randomized the character selection should be
        Output:
            @return: 'n_chars' characters of generated text
        '''
        
        # Begin with seed sequence
        int_encode = seed_int_encode
        
        # Translate seed text
        start_chars = [self.indices_char[x] for x in int_encode]
        message = ''.join(start_chars)
        
        # generate a fixed number of characters
        for _ in range(n_chars):
                        
        	  # truncate sequences to a fixed length
            int_encode = pad_sequences([int_encode],
                                       maxlen=self.seq_len,
                                       truncating='pre')[0]
                        
            # one hot encode
            hot_encode = to_categorical(int_encode,
                                        num_classes=self.vocab_size)
            
            # Change shape from: (seq_len, vocab)
            # to: (1, seq_len, vocab) 
            # Since LSTM requires a tensor input
            hot_encode = np.expand_dims(hot_encode, 0)
            
            # Predict next character
            preds = self.model.predict(hot_encode, verbose=0)[0]
            yhat = self.__sample(preds, diversity)
        
            # Append int encoding to continue recurrant predictions
            int_encode = np.append(int_encode, yhat)
            
            # Keep track of full message generated
            message += self.indices_char[yhat]
            
        # Return generated message
        return(message)
    
    
    
    def __sample(self, preds, temperature=1.0):
        '''
        Allow for diversity to affect which letter is chosen
        
        Input:
            @preds: Probabilities for each letter to be chosen next
            @temperature: How likely higher probabilities are to be chosen
        Output:
            @return:
                Selected letter based on probabilities modified by temperature
        '''
        
        # Ensure probabilities are in the correct format
        preds = np.asarray(preds).astype('float64')
        
        '''
        We log it so that the temperature gets to be a neat value like 
        0.2, 0.5, 1 etc. Otherwise it'd have to be some gross decimal.
        Once we've adjusted based on the temperature, we reverse the nautral
        log with natrual exponent to return them to the pseudo probabilities
        they started as
        '''
        preds = np.log(preds)
        preds /= temperature
        preds = np.exp(preds)
        
        # Convert them to percentages 
        # (So that they sum to 1 and can be used as probabilities)
        total = np.sum(preds)
        preds /= total
        
        # Run 1 experiment, with preds as the probability, repeat this 1 time
        probas = np.random.multinomial(1, preds, 1)
        
        '''
        From the np.random.multinomial documentation:
        
        To throw a dice 20 times, and 20 times again:
    
        >>>np.random.multinomial(20, [1/6.]*6, size=2)
        
        Output:
        array([[3, 4, 3, 3, 4, 3],
               [2, 4, 3, 4, 0, 7]])
        
        For the first run, we threw a 1 three times, a 2 four times, etc.  
        For the second run, we threw a 1 twice, a 2 four times etc.
        
        ------------------------------------------------------------
        
        As we're running this experiment once, our array will be an array of 
        all 0's except for the single 1 which was selected in the one
        experiment.
        
        Therefore np.argmax - or the index where this 1 exists, is the integer
        representation of the character that we have randomized.
        
        eg.
        x = [0, 0, 0, 0, 1, 0, 0]
        >>>np.argmax(x)
        4
        
        indices_char[4] = 'g'
        
        So we have selected 'g' based on our modified probabilities.
        '''
        
        # Return highest probability
        return np.argmax(probas)
        
    
    
    def __load_model_mappings(self):
        '''
        Load a saved LSTM model and the pickled mappings
        
        Output:
            @post:
                - model loaded from file and saved to class
                - {character -> indices} dict loaded and saved to class
                - {indices -> character} dict loaded and saved to class
                - vocabulary size re-determined and saved to class
        '''
        
        # Determine file names from prefix
        model_name = '_'.join([self.prefix, 'model.h5'])
        char_indices_name = '_'.join([self.prefix, 'char_indices.pkl'])
        indices_char_name = '_'.join([self.prefix, 'indices_char.pkl'])
        
        # Load model and mappings
        self.model = keras.models.load_model(model_name)
        self.char_indices = pickle.load(open(char_indices_name, 'rb'))
        self.indices_char = pickle.load(open(indices_char_name, 'rb'))
        
        # Determine vocab size from mappings
        self.vocab_size = len(self.char_indices)
        
        
        
    def generate_text(self, n_chars = 250, diversity = 0.5):
        '''
        Generate random text using the current LSTM model
        
        Input:
            @n_chars: Number of characters to randomly generate
            @diversity: How randomized the character selection should be
        Output:
            @post: Randomly generated text string printed to screen
            @return: Randomly generated text string
        '''
        
        # Get random sequence from training data to use as seed text
        seed_idx = np.random.randint(0, len(self.integer_sequences))
        seed_seq = self.integer_sequences[seed_idx,:]
        
        # Generate random text
        random_text = self.__generate_seq(seed_seq, n_chars, diversity)
        
        # Print and return random text
        print(random_text)
        return(random_text)
        
    






#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                            Code Cemetery
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

'''
def encode_sequences(filename):
    
    # Read all sequences
    full_text = ''
    with open(filename, 'r', encoding="utf-8") as f:
        full_text = f.read()
    
    lines = full_text.split('\n')
    
    # Create mapping so we can encode them as numbers
    chars = sorted(list(set(full_text)))
    mapping = dict((c, i) for i, c in enumerate(chars))
    
    # Map each character to it's integer to create the encoding
    sequences = []
    for line in lines:
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)


    vocab_size = len(mapping)
    print('Vocabulary Size: %d' % vocab_size)
    
    # Split into X and Y (all input vs output character)
    sequences = np.array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    
    # One hot encode the input and output vectors
    sequences = [keras.utils.to_categorical(x, num_classes=vocab_size) for x in X]
    X = np.array(sequences)
    y = keras.utils.to_categorical(y, num_classes=vocab_size)

    # define model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(75, input_shape=(X.shape[1], X.shape[2])))
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))
    print(model.summary())
    
    # compile and fit model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=2)
    
    # save the model and mapping to file
    model.save('model.h5')
    pickle.dump(mapping, open('mapping.pkl', 'wb'))
    
    
    

def load_messages(filePath):
    
    Takes in the filepath for a raw HTML facebook chat output, and converts
    it to a dataframe. Using a regex, pulls out the datetime stamp, user, 
    and text of each messages, saving them as separate columns.
    
    Output:
        @return: A dataframe of the conversation
      
    print("Loading all messages from Facebook...")
    
    # Prepare regex to retrieve messages
    fb_regex = re.compile(r"<div class=\"message\"><div class=\"message_header\"><span class=\"user\">(.*?)</span><span class=\"meta\">(.*?)</span></div></div><p>(.*?)</p>")
    messages = []
    
    # Pull messages out of html format
    with open(filePath, encoding="utf-8") as infile:
        for line in infile:
            new_messages = fb_regex.findall(line)
            messages += new_messages
    
    df = pd.DataFrame(messages, columns = ["person", "date", "text"])
    
    # Convert text to counters
    text = [html.unescape(x) for x in df["text"]]
    df["text"] = text
    
    # Convert to normal date format
    dates = []
    
    # Parse in dates
    for entry in df["date"].values:
        entry = re.sub(":", "", entry)
        date_time = datetime.strptime(entry, "%A, %B %d, %Y at %I%M%p %Z%z")
        ts = pd.Timestamp(date_time)
        dt64 = ts.to_datetime64()
        dates.append(dt64)
    
    # Replace date strings with datetime objects
    df["date"] = dates
    
    # Convert to time series data frame
    df.set_index('date', inplace=True)
    
    # Return formatted dataframe
    return(df)


def create_document(df, filename):
    
    Takes in the dataframe constructed by load_messages as input, and
    saves a nicely formatted version of the conversation to file.
    
    Input:
        @df: Dataframe constructed by load_messages to convert
        @filename: Name of text file to save into
    
    print("Saving messages to text file...")
    
    # Convert to one long string
    messages = []
    pattern = re.compile(r'[^a-z0-9\!\@\#\$\%\^\&\*\(\)\_\-\+\=\/\\[\]\{\}\:\;\"\'\,\<\.\>\? \`\~]')
    for index, row in df.iterrows():
        message = row.text.lower()
        message = re.sub(pattern, '', message)
        messages.append(row.person + ": " + row.text.lower() + '\n')
    
    # Oldest to Newest
    messages.reverse()

    # Save conversation format to text file
    with open(filename, 'w',  encoding="utf-8") as f:
        f.write(''.join(messages))


def necromancy():

    prefix = 'Steve'
    seed = '                                        '
    seed = 'Matthew Hennegan: hey man i just wanted '
    seq_len = len(seed)
    diversity = 0.8
    
    # load the model and mapping
    model_name = '_'.join([prefix, 'model.h5'])
    model = keras.models.load_model(model_name)
    char_indices = pickle.load(open('_'.join([prefix, 'char_indices.pkl']), 'rb'))
    indices_char = pickle.load(open('_'.join([prefix, 'indices_char.pkl']), 'rb'))
    vocab_size = len(char_indices)
    
    text = seed
    int_encoded = [char_indices[char] for char in seed]
    int_encoded = np.array(int_encoded)
    
    # generate a fixed number of characters
    for _ in range(1000):
               
        int_encoded = pad_sequences([int_encoded],
                                maxlen=seq_len,
                                truncating='pre')[0]
        
        
        # one hot encode
        hot_encoded = to_categorical(int_encoded,
                                 num_classes=vocab_size)

        
        hot_encoded = np.expand_dims(hot_encoded, 0)
        
        # Predict model  
        preds = model.predict(hot_encoded, verbose=0)[0]
        yhat = sample(preds, diversity)
    
        # append to input
        int_encoded = np.append(int_encoded, yhat)
        text += indices_char[yhat]
        

    print(text)


def train_LSTM(X, y, vocab_size, prefix):
    
    Takes fully prepared one-hot encoded data, and predicted labels as
    input, and trains an LSTM model on the data.
    
    Input:
        @X: List of hot-encoded Input sequence of chars
        @y: List of hot-encoded outputs from input sequence
        @vocab_size: For how big the one-hot encoded must be
        @prefix: Naming prefix for model name and mappings
    
    # What to call the saved model
    model_name = '_'.join([prefix, 'model.h5'])

    # Define model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(X.shape[1], vocab_size)))
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))
    print(model.summary())
    
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    
    # Create generator and fit model
    data_gen = DataGenerator(X, y, vocab_size)
    
    num_epochs = 100
    for i in range(num_epochs):
        print("Training Epoch: ", i)
        
        # Train next epoch and save
        model.fit_generator(generator = data_gen, epochs = 1, verbose = 1)
        model.save(model_name)
        
        # Test how well the model is currently doing
        seed = data_gen.get_seed_text()
        for j in [0.2, 0.5, 1, 1.2]:
            print("-" * 40)
            print("Testing model with diversity", j)
            print()
            text = generate_incremental_text(model, prefix, seed, 500, j)
            print(text)
            print("-" * 40)
    
    # Return created LSTM model
    return(model)


###############################################################################
#                       Use Model to Generate Text
###############################################################################






# generate a sequence of characters with a language model
def generate_seq(model, char_indices, indices_char, seq_length,
                 seed_text, n_chars, diversity = 0.2):
    
    
    in_text = seed_text
    
    # generate a fixed number of characters
    for _ in range(n_chars):
        
        # encode the characters as integers
        encoded = [char_indices[char] for char in in_text]
    	  # truncate sequences to a fixed length
        encoded = keras.preprocessing.sequence.pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # one hot encode
        encoded = keras.utils.to_categorical(encoded, num_classes=len(char_indices))
    		
        # Predict model  
        preds = model.predict(np.array(encoded,), verbose=0)[0]
        yhat = sample(preds, diversity)
    
        # append to input
        in_text += indices_char[yhat]
        
    return in_text


def generate_text(prefix, seq_length, seed_text,
                  num_chars, diversity = 0.2):

    # load the model and mapping
    model_name = '_'.join([prefix, 'model.h5'])
    model = keras.models.load_model(model_name)
    char_indices = pickle.load(open('_'.join([prefix, 'char_indices.pkl']), 'rb'))
    indices_char = pickle.load(open('_'.join([prefix, 'indices_char.pkl']), 'rb'))
    
    # Generate text
    text = generate_seq(model, char_indices, indices_char, seq_length,
                        seed_text, num_chars, diversity)
    
    # Print and return text
    print(text)
    return(text)


def generate_incremental_text(model, prefix, seed_text,
                              num_chars, diversity):
    
    # load the mapping
    char_indices = pickle.load(open('_'.join([prefix, 'char_indices.pkl']), 'rb'))
    indices_char = pickle.load(open('_'.join([prefix, 'indices_char.pkl']), 'rb'))
    
    # Convert encoded back to text (to be convered back. Yes it's ugly)
    seed_text = [indices_char[ind] for ind in seed_text]
    
    # Generate text
    text = generate_seq(model, char_indices, indices_char, len(seed_text),
                        seed_text, num_chars, diversity)
    
    # Print and return text
    return(text)

###############################################################################
#                               Use Methods
###############################################################################

def create_facebook_model(html_filename, name_prefix = None):
    
    # Convert html to text
    df = load_messages(html_filename)
    
    # Determine prefix to use
    prefix = ''
    if name_prefix is None:
        pat = re.compile('(.*).html')
        prefix = pat.findall(html_filename)[0]
    else:
        prefix = name_prefix
    
    # Save as text file
    text_filename = ''.join([prefix, '.txt'])
    create_document(df, text_filename)

    # Get seqeunces from text file
    X, y, vocab_size = load_encoded_sequences(text_filename, prefix)
    
    # Train LSTM
    train_LSTM(X, y, vocab_size, prefix)
    
    # Return prefix
    return(prefix)



# Create model
#prefix = create_facebook_model('Kevin.html')

# Generate from Model
#seed_text = '                                        '
#seed_text = 'Matthew Hennegan: my babbers one!! :3 :3'
#generate_text(prefix = prefix, seq_length = 40, seed_text = seed_text,
#                   num_chars = 1000, diversity = 0.2)


'''
