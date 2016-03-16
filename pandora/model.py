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
                include_token=True,
                include_context=True,
                include_lemma=True,
                include_pos=True,
                include_morph=True,
                ):
    
    m = Graph()
    subnets = []
    
    if include_token:
        # add input layer:
        m.add_input(name='focus_in',
                    input_shape=(token_len, len(token_char_vector_dict)))

        # add recurrent layers to model focus token:
        for i in range(nb_encoding_layers):
            if i == 0:
                input_name = 'focus_in'
            else:
                input_name = 'encoder_'+str(i)

            if i == (nb_encoding_layers - 1):
                output_name = 'final_focus_encoder'
                m.add_node(LSTM(output_dim=nb_dense_dims,
                                return_sequences=False,
                                activation='tanh'),
                            name=output_name,
                            input=input_name)
            else:
                output_name = 'encoder_'+str(i + 1)
                m.add_node(LSTM(output_dim=nb_dense_dims,
                                return_sequences=True,
                                activation='tanh'),
                            name='encoder_'+str(i + 1),
                            input=input_name)

        subnets.append('final_focus_encoder')

    if include_context:
        # add context embeddings:
        m.add_input(name='context_in',
                    input_shape=(1,),
                    dtype='int')

        # add embedding layers to model context:
        m.add_node(Embedding(input_dim=nb_train_tokens,
                             output_dim=nb_embedding_dims,
                             weights=pretrained_embeddings,
                             input_length=nb_context_tokens),
                       name='context_embedding', input='context_in')
        m.add_node(Flatten(),
                       name="context_flatten", input="context_embedding")
        m.add_node(Dropout(0.5),
                       name='context_dropout', input='context_flatten')
        m.add_node(Activation('relu'),
                       name='context_relu', input='context_dropout')
        m.add_node(Dense(output_dim=nb_dense_dims),
                       name="context_dense1", input="context_relu")
        m.add_node(Dropout(0.5),
                       name="context_dropout2", input="context_dense1")
        m.add_node(Activation('relu'),
                       name='context_out', input='context_dropout2')
        subnets.append('context_out')

    # combine subnets:
    if len(subnets) > 1:
        m.add_node(Activation('linear'),
                   name='joined',
                   inputs=subnets,
                   merge_mode='concat',
                   concat_axis = -1)
    else:
        m.add_node(Activation('linear'),
                   name='joined',
                   input=subnets[0])

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
                input_name = 'decoder_'+str(i)

            if i == (nb_encoding_layers - 1):
                output_name = 'final_focus_decoder'
            else:
                output_name = 'decoder_'+str(i + 1)

            m.add_node(LSTM(output_dim=nb_dense_dims,
                            return_sequences=True,
                            activation='tanh'),
                        name='left_lstm_'+str(i + 1),
                        input=input_name)
            m.add_node(LSTM(output_dim=nb_dense_dims,
                            return_sequences=True,
                            activation='tanh',
                            go_backwards=True),
                        name='right_lstm_'+str(i + 1),
                        input=input_name)
            m.add_node(Activation('linear'),
                        name=output_name,
                        inputs=['left_lstm_'+str(i + 1), 'right_lstm_'+str(i + 1)],
                        merge_mode='sum')

        # add lemma decoder
        m.add_node(TimeDistributedDense(output_dim=len(lemma_char_vector_dict)),
                    name='lemma_dense',
                    input='final_focus_decoder')
        m.add_node(Activation('softmax'),
                    name='lemma_softmax',
                    input='lemma_dense')
        m.add_output(name='lemma_out', input='lemma_softmax')

    if include_pos:
        # add pos tag output:
        m.add_node(Dense(output_dim=nb_dense_dims,
                         activation='relu'),
                   name='pos_dense1',
                   input='joined')
        m.add_node(Dropout(0.15),
                    name='pos_dense_dropout1',
                    input='pos_dense1')
        m.add_node(Dense(output_dim=nb_dense_dims,
                         activation='relu'),
                   name='pos_dense2',
                   input='pos_dense_dropout1')
        m.add_node(Dropout(0.15),
                    name='pos_dense_dropout2',
                    input='pos_dense2')
        m.add_node(Dense(output_dim=nb_tags),
                   name='pos_dense3',
                   input='pos_dense_dropout2')
        m.add_node(Dropout(0.05),
                    name='pos_dense_dropout3',
                    input='pos_dense3')
        m.add_node(Activation('softmax'),
                    name='pos_softmax',
                    input='pos_dense_dropout3')
        m.add_output(name='pos_out', input='pos_softmax')

    if include_morph:
        # add morph tag output:
        m.add_node(Dense(output_dim=nb_morph_cats),
                   name='morph_dense',
                   input='joined')
        m.add_node(Dropout(0.05),
                    name='morph_dense_dropout',
                    input='morph_dense')
        m.add_node(Activation('softmax'),
                    name='morph_softmax',
                    input='morph_dense_dropout')
        m.add_output(name='morph_out', input='morph_softmax')
    
    from keras.objectives import categorical_crossentropy

    loss_dict = {}
    
    if include_lemma:
        loss_dict['lemma_out'] = 'categorical_crossentropy'
    if include_pos:
        #loss_dict['pos_out'] = 'categorical_crossentropy'
        loss_dict['pos_out'] = lambda x, y: 0.2 * categorical_crossentropy(x, y)
    if include_morph:
        loss_dict['morph_out'] = 'categorical_crossentropy'

    m.compile(optimizer='RMSprop', loss=loss_dict)

    return m