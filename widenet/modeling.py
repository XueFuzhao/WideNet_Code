import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model_configs import mlp, Patches, PatchEncoder, Switch, MultiHeadAttention
import configs


def random_mask(ones,drop_rate):
    ones = tf.nn.dropout(ones, rate = drop_rate)
    ones = tf.cast(ones,tf.dtypes.bool)
    ones = tf.cast(ones,tf.dtypes.float32)
    return ones

def create_vit_classifier(config, input_shape, img_size=224, switch_deepth=128, num_classes=1000, use_representation=False, share_att=False, share_ffn=False, group_deepth=128):

    inputs = layers.Input(shape=input_shape)

    # Create patches.
    patch_size = config.patches["size"][0]

    # patches = Patches(patch_size)(augmented)

    patches = layers.Conv2D(config.hidden_size, patch_size, patch_size)(inputs)
    patch_shape = patches.get_shape()
    patches = layers.Reshape((patch_shape[1]*patch_shape[2], patch_shape[3]))(patches)
    # Encode patches.
    num_patches = (img_size // patch_size) ** 2
    # patches = patches * tf.expand_dims(random_mask(tf.ones((num_patches)), 0.15), axis=-1)
    #patches = patches * tf.expand_dims(tf.nn.dropout(tf.ones((num_patches)), rate = 0.1)*(1.0-0.1), axis=-1)
    # shape (batch_size, num_patches, hiddensize)
    encoded_patches = PatchEncoder(num_patches, config.hidden_size, config.transformer["dropout_rate"])(patches)
    # Create multiple layers of the Transformer block.
    
    

    #j=0
    
    for i in range(config.transformer["num_layers"]):
        # Layer normalization 1.
        
        if i % switch_deepth == 0:
            
        
            if share_att:
                atten_normal = layers.LayerNormalization(epsilon=1e-6)
                # Create a multi-head attention layer.
                atten_layer = layers.MultiHeadAttention(
                     num_heads=config.transformer["num_heads"],
                     key_dim=int(config.hidden_size/config.transformer["num_heads"]),
                     value_dim=int(config.hidden_size/config.transformer["num_heads"]),
                     dropout=config.transformer["attention_dropout_rate"]
                 )
                atten_drop = layers.Dropout(config.transformer["dropout_rate"])
                att_add = layers.Add()

            if share_ffn:
                ffn_normal = layers.LayerNormalization(epsilon=1e-6)
                transformer_units = [config.transformer["mlp_dim"], config.hidden_size]
                ffn_mlp = mlp(hidden_units=transformer_units, dropout_rate=config.transformer["dropout_rate"])
                ffn_add = layers.Add()
        
        
        if share_att:
            #x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            x1 = atten_normal(encoded_patches)
            attention_output = atten_layer(x1, x1)
            attention_output = atten_drop(attention_output)
            x2 = att_add([attention_output, encoded_patches])
        else:
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                 num_heads=config.transformer["num_heads"],
                 key_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 value_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 dropout=config.transformer["attention_dropout_rate"]
             )(x1, x1)
            #attention_output = MultiHeadAttention(
            #    num_heads=config.transformer["num_heads"],
            #    head_dim=int(config.hidden_size/config.transformer["num_heads"]),
            #    dropout_rate=config.transformer["attention_dropout_rate"]
            #)(x1)
            # Skip connection 1.
            attention_output = layers.Dropout(config.transformer["dropout_rate"])(attention_output)
            x2 = layers.Add()([attention_output, encoded_patches])
        if share_ffn:
            x3 = ffn_normal(x2)
            #x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = ffn_mlp(x3)
            encoded_patches = ffn_add([x3, x2])
        else:
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            transformer_units = [config.transformer["mlp_dim"], config.hidden_size]
            x3 = mlp(hidden_units=transformer_units, dropout_rate=config.transformer["dropout_rate"])(x3)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])
        
        #j+=1

    # Create a [batch_size, projection_dim] tensor.

    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Average pooling feature
    if config.classifier=='gap':
        features = layers.GlobalAveragePooling1D()(encoded_patches)
        if use_representation:
            features = layers.Dense(config.hidden_size)(features)
            features = layers.Activation('tanh')(features)
    elif config.classifier=='token':
        features = encoded_patches[:, 0]
        if use_representation:
            features = layers.Dense(config.hidden_size)(features)
            features = layers.Activation('tanh')(features)        
    elif config.classifier=='map':
        features = layers.MultiHeadAttention(
                 num_heads=config.transformer["num_heads"],
                 key_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 value_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 dropout=config.transformer["attention_dropout_rate"]
             )(tf.expand_dims(encoded_patches[:, 0],1), encoded_patches)
        features = tf.squeeze(features, [1])
    logits = layers.Dense(num_classes)(features)#,kernel_initializer='zeros')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


def create_vit_moe_classifier(config, input_shape, num_experts,num_masked_experts, batch_size, capacity_factor=1.0,
                              top_k=1,aux_loss=True, aux_loss_alpha=1.0,aux_loss_beta=0.001, img_size=224,
                              switch_deepth=1,
                              num_classes=1000,use_representation=False,share_att=False,share_ffn=False, 
                              group_deepth = 128):
    
    inputs = layers.Input(shape=input_shape)

    # Create patches.
    patch_size = config.patches["size"][0]

    # patches = Patches(patch_size)(augmented)

    patches = layers.Conv2D(config.hidden_size, patch_size, patch_size)(inputs)
    patch_shape = patches.get_shape()
    patches = layers.Reshape((patch_shape[1]*patch_shape[2], patch_shape[3]))(patches)
    # Encode patches.
    num_patches = (img_size // patch_size) ** 2
    #patches = patches * tf.expand_dims(random_mask(tf.ones((num_patches)), 0.15), axis=-1)
    # shape (batch_size, num_patches, hiddensize)
    encoded_patches = PatchEncoder(num_patches, config.hidden_size, config.transformer["dropout_rate"])(patches)
    num_patches = num_patches+1

    
    for i in range(config.transformer["num_layers"]):
        
        if i % switch_deepth == 0:
            if share_att:
                #atten_normal = layers.LayerNormalization(epsilon=1e-6)
                # Create a multi-head attention layer.
                atten_layer = layers.MultiHeadAttention(
                     num_heads=config.transformer["num_heads"],
                     key_dim=int(config.hidden_size/config.transformer["num_heads"]),
                     value_dim=int(config.hidden_size/config.transformer["num_heads"]),
                     dropout=config.transformer["attention_dropout_rate"]
                 )
                atten_drop = layers.Dropout(config.transformer["dropout_rate"])
                att_add = layers.Add()

            if share_ffn:
                #ffn_normal = layers.LayerNormalization(epsilon=1e-6)
                if i < config.transformer["num_layers"]-switch_deepth:
                    transformer_units = [config.transformer["mlp_dim"], config.hidden_size]
                    base_ffn_mlp = mlp(hidden_units=transformer_units, dropout_rate=config.transformer["dropout_rate"])
                else:
                    ffn_mlp = Switch(config, num_patches, num_experts, num_masked_experts, batch_size, capacity_factor,
                                 top_k=top_k, switch_deepth=switch_deepth, group_deepth = group_deepth, aux_loss=aux_loss, 
                                 aux_loss_alpha=aux_loss_alpha,
                                 aux_loss_beta=aux_loss_beta)
                ffn_add = layers.Add()
        
        # Layer normalization 1.
        if share_att:
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            #x1 = atten_normal(encoded_patches)
            attention_output = atten_layer(x1, x1)
            attention_output = atten_drop(attention_output)
            x2 = att_add([attention_output, encoded_patches])
        else:
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                 num_heads=config.transformer["num_heads"],
                 key_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 value_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 dropout=config.transformer["attention_dropout_rate"]
             )(x1, x1)
            #attention_output = MultiHeadAttention(
            #    num_heads=config.transformer["num_heads"],
            #    head_dim=int(config.hidden_size/config.transformer["num_heads"]),
            #    dropout_rate=config.transformer["attention_dropout_rate"]
            #)(x1)
            # Skip connection 1.
            attention_output = layers.Dropout(config.transformer["dropout_rate"])(attention_output)
            x2 = layers.Add()([attention_output, encoded_patches])
        if share_ffn:
            #x3 = ffn_normal(x2)
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            if i < config.transformer["num_layers"]-switch_deepth:
            #if i%switch_deepth==0:
                x3 = base_ffn_mlp(x3)
            else:
                x3 = ffn_mlp(x3)
            encoded_patches = ffn_add([x3, x2])
        else:
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            #x3 = ffn_normal
            # MLP.
            transformer_units = [config.transformer["mlp_dim"], config.hidden_size]
            if i < config.transformer["num_layers"]-switch_deepth:
            #if i%switch_deepth==0:
                x3 = mlp(hidden_units=transformer_units, dropout_rate=config.transformer["dropout_rate"])(x3)
            else:
                x3 = Switch(config, num_patches, num_experts, num_masked_experts, 
                                                            batch_size, capacity_factor,
                                                            top_k=top_k, switch_deepth=switch_deepth,
                                                            group_deepth = group_deepth,
                                                            aux_loss=aux_loss, aux_loss_alpha=aux_loss_alpha,
                                                            aux_loss_beta=aux_loss_beta)(x3)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])
            


    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Average pooling feature
    if config.classifier=='gap':
        features = layers.GlobalAveragePooling1D()(encoded_patches)
        if use_representation:
            features = layers.Dense(config.hidden_size)(features)
            features = layers.Activation('tanh')(features)
    elif config.classifier=='token':
        features = encoded_patches[:, 0]
        if use_representation:
            features = layers.Dense(config.hidden_size)(features)
            features = layers.Activation('tanh')(features)        
    elif config.classifier=='map':
        features = layers.MultiHeadAttention(
                 num_heads=config.transformer["num_heads"],
                 key_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 value_dim=int(config.hidden_size/config.transformer["num_heads"]),
                 dropout=config.transformer["attention_dropout_rate"]
             )(tf.expand_dims(encoded_patches[:, 0],1), encoded_patches)
        features = tf.squeeze(features, [1])
    logits = layers.Dense(num_classes)(features)#,kernel_initializer='zeros')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model



CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'ViT-XH_14': configs.get_xh14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'ViT-MoE-B_16': configs.get_moeb16_config(),
    'ViT-MoE-L_16': configs.get_moel16_config(),
    'ViT-MoE-H_14': configs.get_moeh14_config(),
    'ViT-MoE-XH_14': configs.get_moexh14_config(),
    'testing': configs.get_testing()
}
