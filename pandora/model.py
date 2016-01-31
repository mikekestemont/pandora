#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Graph
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, TimeDistributedDense,\
                              Dropout, Activation, RepeatVector,\
                              Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

def build_model(token_len, token_char_vector_dict,
                nb_encoding_layers, nb_dense_dims,
                lemma_len, lemma_char_vector_dict,
                nb_tags, nb_morph_cats, nb_train_tokens,
                nb_left_tokens, nb_right_tokens,
                nb_embedding_dims,
                pretrained_embeddings=None,
                ):
    
    m = Graph()

    # add input layer:
    m.add_input(name='focus_in',
                input_shape=(token_len, len(token_char_vector_dict)))

    # add context embeddings:
    m.add_input(name='left_in',
                input_shape=(1,),
                dtype='int')
    m.add_input(name='right_in', 
                input_shape=(1,),
                dtype='int')

    # add recurrent layers for focus token:
    return_seqs = True
    for i in range(nb_encoding_layers):
        if i == 0:
            input_name = 'focus_in'
        else:
            input_name = 'encoder_dropout_'+str(i)

        if i == (nb_encoding_layers - 1):
            output_name = 'final_focus_encoder'
            return_seqs = False
        else:
            output_name = 'encoder_dropout_'+str(i + 1)

        m.add_node(LSTM(input_dim=nb_dense_dims,
                        output_dim=nb_dense_dims,
                        return_sequences=return_seqs,
                        activation='tanh'),
                        name='encoder_'+str(i + 1),
                        input=input_name)
        m.add_node(Dropout(0.05),
                    name=output_name,
                    input='encoder_'+str(i + 1))

    
    # add contextual embedding layers:
    m.add_node(Embedding(input_dim=nb_train_tokens,
                         output_dim=nb_embedding_dims,
                         weights=pretrained_embeddings,
                         input_length=nb_left_tokens),
                   name='left_embedding', input='left_in')
    m.add_node(Flatten(),
                   name="left_flatten", input="left_embedding")
    m.add_node(Dropout(0.5),
                   name='left_dropout', input='left_flatten')
    m.add_node(Activation('relu'),
                   name='left_relu', input='left_dropout')
    m.add_node(Dense(output_dim=nb_dense_dims),
                   name="left_dense1", input="left_relu")
    m.add_node(Dropout(0.5),
                   name="left_dropout2", input="left_dense1")
    m.add_node(Activation('relu'),
                   name='left_out', input='left_dropout2')

    m.add_node(Embedding(input_dim=nb_train_tokens,
                         output_dim=nb_embedding_dims,
                         weights=pretrained_embeddings,
                         input_length=nb_right_tokens),
                   name='right_embedding', input='right_in')
    m.add_node(Flatten(),
                   name="right_flatten", input="right_embedding")
    m.add_node(Dropout(0.5),
                   name='right_dropout', input='right_flatten')
    m.add_node(Activation('relu'),
                   name='right_relu', input='right_dropout')
    m.add_node(Dense(output_dim=nb_dense_dims),
                   name="right_dense1", input="right_relu")
    m.add_node(Dropout(0.5),
                   name="right_dropout2", input="right_dense1")
    m.add_node(Activation('relu'),
                   name='right_out', input='right_dropout2')
    
    m.add_node(Activation('linear'),
               name='joined',
               inputs=['left_out', 'final_focus_encoder', 'right_out'],
               merge_mode='concat',
               concat_axis = -1)
    # repeat final input
    m.add_node(RepeatVector(lemma_len),
          name='encoder_repeat',
          input='joined')
    
    # add recurrent layers to generate lemma:
    for i in range(nb_encoding_layers):
        if i == 0:
            input_name = 'encoder_repeat'
        else:
            input_name = 'decoder_dropout_'+str(i)

        if i == (nb_encoding_layers - 1):
            output_name = 'final_focus_decoder'
        else:
            output_name = 'decoder_dropout_'+str(i + 1)

        m.add_node(LSTM(input_dim=nb_dense_dims,
                        output_dim=nb_dense_dims,
                        return_sequences=True,
                        activation='tanh'),
                        name='decoder_'+str(i + 1),
                        input=input_name)
        m.add_node(Dropout(0.05),
                    name=output_name,
                    input='decoder_'+str(i + 1))
    """
    # 2nd, single recurrent layer to generate output sequence:
    m.add_node(LSTM(input_dim=nb_dense_dims,
                    output_dim=nb_dense_dims,
                    activation='tanh',
                    return_sequences=True),
           input='encoder_repeat',
           name='decoder')
    m.add_node(Dropout(0.05),
                name='final_focus_decoder',
                input='decoder')
    """
    

    # add lemma decoder
    m.add_node(TimeDistributedDense(output_dim=len(lemma_char_vector_dict)),
                name='lemma_dense',
                input='final_focus_decoder')
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
               input='final_focus_encoder')
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
               inputs=['final_focus_encoder', 'pos_softmax'])
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