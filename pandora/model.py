#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam, RMSprop

def build_model(token_len, token_char_vector_dict,
                nb_encoding_layers, nb_dense_dims,
                lemma_len, lemma_char_vector_dict,
                nb_tags, nb_morph_cats,
                nb_lemmas, nb_train_tokens,
                nb_context_tokens,
                nb_embedding_dims,
                pretrained_embeddings=None,
                include_token=True,
                include_context=True,
                include_lemma=True,
                include_pos=True,
                include_morph=True,
                nb_filters = 100,
                filter_length = 3,
                focus_repr = 'recurrent',
                dropout_level = .15,
                ):
    
    inputs, outputs = [], []
    subnets = []
    
    if include_token:
        # add input layer:
        token_input = Input(shape=(token_len, len(token_char_vector_dict)),
                            name='focus_in')
        inputs.append(token_input)

        if focus_repr == 'recurrent':
            # add recurrent layers to model focus token:
            for i in range(nb_encoding_layers):
                if i == 0:
                    curr_input = token_input
                else:
                    curr_input = curr_enc_out

                if i == (nb_encoding_layers - 1):
                    token_subnet = Bidirectional(LSTM(output_dim=nb_dense_dims,
                                                  return_sequences=False,
                                                  activation='tanh',
                                                  name='final_focus_encoder'),
                                             merge_mode='sum')(curr_input)
                else:
                    curr_enc_out = Bidirectional(LSTM(output_dim=nb_dense_dims,
                                                  return_sequences=True,
                                                  activation='tanh',
                                                  name='encoder_'+str(i+1)),
                                             merge_mode='sum')(curr_input)
        elif focus_repr == 'convolutions':
            token_subnet = Convolution1D(input_shape=(token_len, len(token_char_vector_dict)),
                                         nb_filter=nb_filters,
                                         filter_length=filter_length,
                                         activation='relu',
                                         border_mode='valid',
                                         subsample_length=1,
                                         init='glorot_uniform',
                                         name='focus_conv')(token_input)
            token_subnet = Flatten(name='focus_flat')(token_subnet)
            token_subnet = Dropout(dropout_level, name='focus_dropout1')(token_subnet)
            token_subnet = Dense(nb_dense_dims, name='focus_dense')(token_subnet)
            token_subnet = Dropout(dropout_level, name='focus_dropout2')(token_subnet)
            token_subnet = Activation('relu', name='final_focus_encoder')(token_subnet)

        else:
            raise ValueError('Parameter `focus_repr` not understood: use "recurrent" or "convolutions".')

        subnets.append(token_subnet)

    if include_context:
        context_input = Input(shape=(nb_context_tokens,), dtype='int32', name='context_in')
        inputs.append(context_input)

        context_subnet = Embedding(input_dim=nb_train_tokens,
                             output_dim=nb_embedding_dims,
                             weights=pretrained_embeddings,
                             input_length=nb_context_tokens,
                             name='context_embedding')(context_input)
        context_subnet = Flatten(name='context_flatten')(context_subnet)
        context_subnet = Dropout(dropout_level, name='context_dropout')(context_subnet)
        context_subnet = Activation('relu', name='context_relu')(context_subnet)
        context_subnet = Dense(nb_dense_dims, name='context_dense1')(context_subnet)
        context_subnet = Dropout(dropout_level, name='context_dropout2')(context_subnet)
        context_subnet = Activation('relu', name='context_out')(context_subnet)

        subnets.append(context_subnet)

    # combine subnets:
    if len(subnets) > 1:
        joined = merge(subnets, mode='concat', name='joined')
    else:
        joined = Activation('linear', name='joined')(subnets[0])

    if include_lemma:
        if include_lemma == 'generate':
            repeat = RepeatVector(lemma_len, name='encoder_repeat')(joined)

            for i in range(nb_encoding_layers):
                if i == 0:
                    curr_input = repeat
                else:
                    curr_input = curr_out

                if i == (nb_encoding_layers - 1):
                    output_name = 'final_focus_decoder'
                else:
                    output_name = 'decoder_'+str(i + 1)

                curr_out = Bidirectional(LSTM(output_dim=nb_dense_dims,
                                                  return_sequences=True,
                                                  activation='tanh',
                                                  name=output_name),
                                             merge_mode='sum')(curr_input)
            # add lemma decoder
            lemma_label = TimeDistributed(Dense(len(lemma_char_vector_dict)),
                            name='lemma_dense')(curr_out)
            lemma_label = Activation('softmax', name='lemma_out')(lemma_label)

        elif include_lemma == 'label':
            lemma_label = Dense(nb_lemmas,
                                name='lemma_dense1')(joined)
            lemma_label = Dropout(dropout_level,
                                name='lemma_dense_dropout1')(lemma_label)
            lemma_label = Activation('softmax',
                                name='lemma_out')(lemma_label)

        outputs.append(lemma_label)

    if include_pos:
        pos_label = Dense(nb_tags,
                            name='pos_dense1')(joined)
        pos_label = Dropout(dropout_level,
                            name='pos_dense_dropout1')(pos_label)
        pos_label = Activation('softmax',
                            name='pos_out')(pos_label)
        outputs.append(pos_label)

    if include_morph:
        if include_morph == 'label':
            morph_label = Dense(nb_dense_dims,
                                activation='relu',
                                name='morph_dense1')(joined)
            morph_label = Dropout(dropout_level,
                                name='morph_dense_dropout1')(morph_label)
            morph_label = Dense(nb_dense_dims,
                                activation='relu',
                                name='morph_dense2')(morph_label)
            morph_label = Dropout(dropout_level,
                                name='morph_dense_dropout2')(morph_label)
            morph_label = Dense(nb_morph_cats,
                                activation='relu',
                                name='morph_dense3')(morph_label)
            morph_label = Dropout(dropout_level,
                                name='morph_dense_dropout3')(morph_label)
            morph_label = Activation('softmax',
                                    name='morph_out')(morph_label)

        elif include_morph == 'multilabel':
            morph_label = Dense(nb_dense_dims,
                                activation='relu',
                                name='morph_dense1')(joined)
            morph_label = Dropout(dropout_level,
                                name='morph_dense_dropout1')(morph_label)
            morph_label = Dense(nb_dense_dims,
                                activation='relu',
                                name='morph_dense2')(morph_label)
            morph_label = Dropout(dropout_level,
                                name='morph_dense_dropout2')(morph_label)
            morph_label = Dense(nb_morph_cats,
                                activation='relu',
                                name='morph_dense3')(morph_label)
            morph_label = Dropout(dropout_level,
                                name='morph_dense_dropout3')(morph_label)
            morph_label = Activation('tanh',
                                    name='morph_out')(morph_label)

        outputs.append(morph_label)

    loss_dict = {}
    if include_lemma:
        loss_dict['lemma_out'] = 'categorical_crossentropy'
    if include_pos:
        loss_dict['pos_out'] = 'categorical_crossentropy'
    if include_morph:
        if include_morph == 'label':
          loss_dict['morph_out'] = 'categorical_crossentropy'
        elif include_morph == 'multilabel':
          loss_dict['morph_out'] = 'binary_crossentropy'
    
    model = Model(input=inputs, output=outputs)
    if focus_repr == 'convolutions':
        model.compile(optimizer='SGD', loss=loss_dict)
    else:
        model.compile(optimizer='RMSprop', loss=loss_dict)
    
    return model