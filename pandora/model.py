#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Graph
from keras.layers.recurrent import LSTM
from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam

def build_model(token_len, token_char_vector_dict,
                nb_encoding_layers, nb_dense_dims,
                lemma_len, lemma_char_vector_dict,
                nb_tags, nb_morph_cats, nb_train_tokens,
                nb_context_tokens,
                nb_embedding_dims,
                pretrained_embeddings=None,
                include_lemma=True,
                include_pos=True,
                include_morph=True,
                ):
    
    m = Graph()

    # add input layer:
    m.add_input(name='focus_in',
                input_shape=(token_len, len(token_char_vector_dict)))

    # add context embeddings:
    m.add_input(name='context_in',
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
        m.add_node(Dropout(0.01),
                    name=output_name,
                    input='encoder_'+str(i + 1))

    
    # add contextual embedding layers:
    m.add_node(Embedding(input_dim=nb_train_tokens,
                         output_dim=nb_embedding_dims,
                         weights=pretrained_embeddings,
                         input_length=nb_context_tokens),
                   name='context_embedding', input='context_in')
    m.add_node(Flatten(),
                   name="context_flatten", input="context_embedding")
    m.add_node(Dropout(0.25),
                   name='context_dropout', input='context_flatten')
    m.add_node(Activation('relu'),
                   name='context_relu', input='context_dropout')
    m.add_node(Dense(output_dim=nb_dense_dims),
                   name="context_dense1", input="context_relu")
    m.add_node(Dropout(0.25),
                   name="context_dropout2", input="context_dense1")
    m.add_node(Activation('relu'),
                   name='context_out', input='context_dropout2')
    
    # join subnets:
    m.add_node(Activation('linear'),
               name='joined',
               inputs=['final_focus_encoder', 'context_out'],
               merge_mode='concat',
               concat_axis = -1)

    if include_lemma:
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
            m.add_node(Dropout(0.01),
                        name=output_name,
                        input='decoder_'+str(i + 1))
        # add lemma decoder
        m.add_node(TimeDistributedDense(output_dim=len(lemma_char_vector_dict)),
                    name='lemma_dense',
                    input='final_focus_decoder')
        m.add_node(Dropout(0.01),
                    name='lemma_dense_dropout',
                    input='lemma_dense')
        m.add_node(Activation('softmax'),
                    name='lemma_softmax',
                    input='lemma_dense_dropout')
        m.add_output(name='lemma_out', input='lemma_softmax')

    if include_pos:
        # add pos tag output:
        m.add_node(Dense(output_dim=nb_tags),
                   name='pos_dense',
                   input='joined')
        m.add_node(Dropout(0.25),
                    name='pos_dense_dropout',
                    input='pos_dense')
        m.add_node(Activation('softmax'),
                    name='pos_softmax',
                    input='pos_dense_dropout')
        m.add_output(name='pos_out', input='pos_softmax')

    if include_morph:
        # add morph tag output:
        m.add_node(Dense(output_dim=nb_morph_cats),
                   name='morph_dense',
                   input='joined')
        m.add_node(Dropout(0.25),
                    name='morph_dense_dropout',
                    input='morph_dense')
        m.add_node(Activation('softmax'),
                    name='morph_softmax',
                    input='morph_dense_dropout')
        m.add_output(name='morph_out', input='morph_softmax')
    
    loss_dict = {}
    if include_lemma:
        loss_dict['lemma_out'] = 'categorical_crossentropy'
    if include_pos:
        loss_dict['pos_out'] = 'categorical_crossentropy'
    if include_morph:
        loss_dict['morph_out'] = 'categorical_crossentropy'

    adam = Adam(epsilon=1e-8, clipnorm=5)
    m.compile(optimizer=adam, loss=loss_dict)

    return m