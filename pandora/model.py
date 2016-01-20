#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Graph
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, TimeDistributedDense, Dropout, Activation, RepeatVector
from keras.optimizers import Adam

def build_model(token_len, token_char_vector_dict,
                nb_encoding_layers, nb_dense_dims,
                lemma_len, lemma_char_vector_dict,
                nb_tags, nb_morph_cats,
                ):
    m = Graph()

    # add input layer:
    m.add_input(name='focus_in',
                input_shape=(token_len, len(token_char_vector_dict)))

    # add recurrent layers:
    return_seqs = True
    for i in range(nb_encoding_layers):
        if i == 0:
            input_name = 'focus_in'
        else:
            input_name = 'encoder_dropout_'+str(i)

        if i == nb_encoding_layers-1:
            output_name = 'final_encoder'
            return_seqs = False
        else:
            output_name = 'encoder_dropout_'+str(i+1)

        m.add_node(LSTM(output_dim=nb_dense_dims,
                        return_sequences=return_seqs,
                        activation='tanh'),
                        name='encoder_'+str(i+1),
                        input=input_name)
        m.add_node(Dropout(0.05),
                    name=output_name,
                    input='encoder_'+str(i+1))

    # repeat final input
    m.add_node(RepeatVector(lemma_len),
          name='encoder_repeat',
          input='final_encoder')

    # 2nd, single recurrent layer to generate output sequence:
    m.add_node(LSTM(input_dim=nb_dense_dims,
                    output_dim=nb_dense_dims,
                    activation='tanh',
                    return_sequences=True),
           input='encoder_repeat',
           name='decoder')
    m.add_node(Dropout(0.05),
                name='decoder_dropout',
                input='decoder')

    # add lemma decoder
    m.add_node(TimeDistributedDense(output_dim=len(lemma_char_vector_dict)),
                name='lemma_dense',
                input='decoder_dropout')
    m.add_node(Dropout(0.05),
                name='lemma_dense_dropout',
                input='lemma_dense')
    m.add_node(Activation('softmax'),
                name='lemma_softmax',
                input='lemma_dense_dropout')
    m.add_output(name='lemma_out', input='lemma_softmax')

    # add pos tag output:
    m.add_node(Dense(output_dim=nb_tags),
               name='pos_dense',
               input='final_encoder')
    m.add_node(Dropout(0.05),
                name='pos_dense_dropout',
                input='pos_dense')
    m.add_node(Activation('softmax'),
                name='pos_softmax',
                input='pos_dense_dropout')
    m.add_output(name='pos_out', input='pos_softmax') 

    
    # add morph-analysis output:
    m.add_node(Dense(output_dim=nb_morph_cats),
               name='morph_dense',
               inputs=['final_encoder', 'pos_softmax'])
    m.add_node(Dropout(0.05),
                name='morph_dense_dropout',
                input='morph_dense')
    m.add_node(Activation('sigmoid'),
                name='morph_sigmoid',
                input='morph_dense_dropout')
    m.add_output(name='morph_out', input='morph_sigmoid')        
    

    adam = Adam(epsilon=1e-8, clipnorm=5)

    m.compile(optimizer=adam,
              loss={
                    'lemma_out': 'categorical_crossentropy',
                    'pos_out': 'categorical_crossentropy',
                    'morph_out': 'binary_crossentropy',
                    })

    return m