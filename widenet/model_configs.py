import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import math
import six
import ml_collections


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def get_activation(identifier):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.
    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf.keras.activations.get.
    Args:
        identifier: String name of the activation function or callable.
    Returns:
        A Python function corresponding to the activation function.
    """
    if isinstance(identifier, six.string_types):
        name_to_fn = {"gelu": gelu}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


# Multi Head Attention
class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, head_dim, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = num_heads*head_dim
        self.dropout_rate = dropout_rate

        self.query = layers.Dense(units=self.hidden_size)
        self.key = layers.Dense(units=self.hidden_size)
        self.value = layers.Dense(units=self.hidden_size)
        self.out = layers.Dense(units=self.hidden_size)

        self.attn_dropout = layers.Dropout(dropout_rate)
        self.proj_dropout = layers.Dropout(dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = [tf.shape(x)[0], x.shape[1], self.num_heads, self.head_dim]
        x = tf.reshape(x, new_x_shape)
        y = layers.Permute((2,1,3))(x)
        return y

    def call(self, hidden_states):
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(hidden_states)
        mixed_value = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query)
        key_layer = self.transpose_for_scores(mixed_key)
        value_layer = self.transpose_for_scores(mixed_value)

        attention_scores = tf.linalg.matmul(query_layer,
                                            tf.transpose(key_layer, (0, 1, 3, 2)))
        attention_scores = attention_scores/math.sqrt(self.head_dim)
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = tf.linalg.matmul(attention_probs, value_layer)
        context_layer = layers.Permute((2, 1, 3))(context_layer)
        new_context_layer_shape = [tf.shape(context_layer)[0], context_layer.shape[1], self.hidden_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output




# Two layer multilayer perceptron (MLP)
def mlp(hidden_units, dropout_rate, name=None):
    return keras.Sequential(
        [layers.Dense(hidden_units[0], activation=get_activation('gelu'),
                      bias_initializer=keras.initializers.RandomNormal(stddev=1e-6)),
         layers.Dropout(dropout_rate),
         layers.Dense(hidden_units[1], activation=get_activation('gelu'),
                      bias_initializer=keras.initializers.RandomNormal(stddev=1e-6)),
         layers.Dropout(dropout_rate)],
        name=name
    )


# patch creation
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, dropout_rate):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches+1, output_dim=projection_dim,
                      embeddings_initializer=keras.initializers.RandomNormal(stddev=0.02))
        self.cls_token = tf.zeros((1,1, projection_dim))
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches+1,delta=1)#+1, delta=1)
        shape_tensor = tf.expand_dims(layers.GlobalAveragePooling1D()(patch), axis=1)
        cls_token = tf.zeros_like(shape_tensor)
        patch = tf.concat([cls_token, patch], axis=1)
        #encoded = self.projection(patch) + self.position_embedding(positions)
        encoded = patch + self.position_embedding(positions)
        encoded = self.dropout(encoded)
        return encoded

    def get_config(self):
        config = {
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
            'dropout_rate': self.dropout_rate
        }
        
        return config


# An auxiliary loss to encourage a balanced loss across experts
def load_balanced_loss(router_probs, expert_mask, loss_alpha=0.01):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = tf.shape(expert_mask)[-1]
    # Get the fraction of tokens routed to each expert.
    # density is a vector of length num experts that sums to 1.
    density = tf.reduce_mean(expert_mask, axis=0)
    # Get fraction of probability mass assigned to each expert from the router
    # across all tokens. density_proxy is a vector of length num experts that sums to 1.
    density_proxy = tf.reduce_mean(router_probs, axis=0)
    # Want both vectors to have uniform allocation (1/num experts) across all
    # num_expert elements. The two vectors will be pushed towards uniform allocation
    # when the dot product is minimized.
    loss = tf.reduce_mean(density_proxy * density) * tf.cast(
        (num_experts ** 2), tf.dtypes.float32
    )
    return tf.cast(loss*loss_alpha, tf.dtypes.float32)

def sliding_loss(px, qx, loss_alpha=0.001):

    #loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(px-qx,2), axis=1)))
    loss = tf.reduce_mean(tf.reduce_sum(tf.abs(px-qx), axis=1))
    #loss = tf.reduce_mean(tf.reduce_sum(x, axis=1), axis=0)

    return tf.cast(loss*loss_alpha, tf.dtypes.float32)

def kl_sliding_loss(px, qx, loss_alpha=0.001):
    loss = tf.reduce_sum(px*(tf.math.log(px)-tf.math.log(qx)))
    #loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.pow(px-qx,2), axis=1)))
    #loss = tf.reduce_mean(tf.reduce_sum(x, axis=1), axis=0)

    return tf.cast(-1.0*loss*loss_alpha, tf.dtypes.float32)

def importance_loss(router_probs, loss_alpha=0.01):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = tf.shape(router_probs)[-1]
    density = tf.reduce_mean(router_probs, axis=0)
    loss = tf.math.reduce_variance(density)

    return tf.cast(loss*loss_alpha, tf.dtypes.float32)

def load_loss(router_probs, loss_alpha=0.01):
    # router_probs [tokens_per_batch, num_experts] is the probability assigned for
    # each expert per token. expert_mask [tokens_per_batch, num_experts] contains
    # the expert with the highest router probability in one−hot format.

    num_experts = tf.shape(router_probs)[-1]
    density = tf.reduce_mean(router_probs, axis=0)
    loss = tf.reduce_variance(density)

    return tf.cast(loss*loss_alpha, tf.dtypes.float32)

def random_mask(ones,drop_rate):
    ones = tf.nn.dropout(ones, rate = drop_rate)
    ones = tf.cast(ones,tf.dtypes.bool)
    ones = tf.cast(ones,tf.dtypes.float32)
    return ones


# Router for Switch layer
class Router(layers.Layer):
    def __init__(self, num_experts, num_masked_experts, expert_capacity, top_k=1,aux_loss=True,aux_loss_alpha=1.0,num_tokens_per_sample=1):
        self.top_k = top_k #tf.cast(top_k, tf.dtypes.int32)
        self.num_masked_experts = num_masked_experts
        self.num_experts = num_experts #tf.cast(num_experts, tf.dtypes.int32)
        self.route = layers.Dense(units=num_experts,use_bias=False)
        self.expert_capacity_int = int(expert_capacity)
        self.expert_capacity = tf.cast(expert_capacity, tf.dtypes.int32)
        #self.fixed_capacity = fixed_capacity
        self.aux_loss = aux_loss
        self.aux_loss_alpha = aux_loss_alpha
        self.num_tokens_per_sample = num_tokens_per_sample
        self.expert_dropout_ones = tf.ones((num_experts))
        #self.num_tokens_per_batch = num_tokens_per_batch
        super(Router, self).__init__()

    def call(self, inputs, training=False):
        # inputs shape: [tokens_per_batch, embed_dim]
        # router_logits shape: [tokens_per_batch, num_experts]
        #num_tokens_per_batch = tf.shape(inputs)[0]
        router_logits = self.route(inputs)

        if training:
            # Add noise for exploration across experts.

            router_logits += tf.random.normal(
                shape=tf.shape(router_logits), mean=0.0, stddev=1.0/(self.num_experts) # ** 2)
            )
        # Probabilities for each token of what expert it should be sent to.
        #router_mask = (random_mask(self.expert_dropout_ones, self.num_masked_experts/self.num_experts)-1.0)*1e+9
        #router_logits = router_logits * tf.expand_dims(router_mask, axis=0)
        router_probs = keras.activations.softmax(router_logits, axis=-1)
        
        #if self.aux_loss:
        #    aux_loss = importance_loss(router_probs, self.aux_loss_alpha)
        #    self.add_loss(aux_loss)
        # router_mask = (random_mask(self.expert_dropout_ones, rate = self.num_masked_experts/self.num_experts)-1.0)*1e+9
        # router_probs = router_probs * tf.expand_dims(router_mask, axis=0)
        # Get the top−1 expert for each token. expert_gate is the top−1 probability
        # from the router for each token. expert_index is what expert each token
        # is going to be routed to.
        expert_gate, expert_index = tf.math.top_k(router_probs, k=self.top_k)
        
        if self.top_k==1:
            expert_index = tf.squeeze(expert_index, [1])
            expert_gate = tf.squeeze(expert_gate, [1])
            #expert_mask shape: [num_tokens_per_batch, num_experts]
            expert_mask = tf.one_hot(expert_index, depth=self.num_experts)
            # Compute load balancing loss.
            if self.aux_loss:
                aux_loss = load_balanced_loss(router_probs, expert_mask,self.aux_loss_alpha)
                self.add_loss(aux_loss)
            # Shape of position_in_expert: [tokens_per_batch, num_experts]
            position_in_expert = tf.cast(
                tf.math.cumsum(expert_mask, axis=0) * expert_mask, tf.dtypes.int32
            )
            # Keep only tokens that fit within expert capacity.
            # Experts have a fixed capacity, ensure we do not exceed it. Construct
            # the batch indices, to each expert, with position in expert make sure that
            # not more that expert capacity examples can be routed to each expert.
            expert_mask *= tf.cast(
                tf.math.less(position_in_expert, self.expert_capacity),tf.dtypes.float32,
            )
            # expert_mask_flat shape: [tokens_per_batch]
            expert_mask_flat = tf.reduce_sum(expert_mask, axis=-1)
            # Mask out the experts that have overflowed the expert capacity.
            expert_gate *= expert_mask_flat
            # Combine expert outputs and scaling with router probability.
            # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
            combined_tensor = tf.expand_dims(tf.expand_dims(expert_gate * expert_mask_flat, axis=-1) * tf.one_hot(expert_index,depth=self.num_experts), axis=-1) * tf.one_hot(position_in_expert, depth=self.expert_capacity_int)
            # Create binary dispatch_tensor [tokens_per_batch, num_experts, expert_capacity]
            # that is 1 if the token gets routed to the corresponding expert.
            #dispatch_tensor = tf.cast(combined_tensor, tf.dtypes.float32)
            dispatch_tensor = tf.cast(combined_tensor,tf.dtypes.bool)
            dispatch_tensor = tf.cast(dispatch_tensor, tf.dtypes.float32)
            
            
            
        elif self.top_k>1: 
            #expert_gate_list = []
            expert_mask_list = []
            for i in range(self.top_k):
                #expert_gate_list.append(expert_gate[:,i])#tf.reshape(,shape=[-1])
                #expert_mask_sum shape: [tokens_per_batch, num_experts]
                expert_mask_list.append(tf.one_hot(expert_index[:,i], depth=self.num_experts))
                
            expert_mask_sum = tf.reduce_sum(tf.one_hot(expert_index, depth=self.num_experts), axis=1)
            # Compute load balancing loss.
            if self.aux_loss:
                aux_loss = load_balanced_loss(router_probs,
                                              tf.cast(expert_mask_sum/self.top_k,tf.dtypes.float32) ,
                                              self.aux_loss_alpha)
                self.add_loss(aux_loss)

            
            # expert_mask_list_stacked shape: [top_k, tokens_per_batch, num_experts]
            
            position_in_expert = tf.cast(
                tf.math.cumsum(expert_mask_sum, axis=0) * expert_mask_sum, tf.dtypes.int32
            )
            expert_mask_sum *= tf.cast(
                tf.math.less(position_in_expert, self.expert_capacity),tf.dtypes.float32,
            )
            expert_mask_flat = tf.cast(tf.cast(tf.reduce_sum(expert_mask_sum, axis=-1),tf.dtypes.bool),tf.dtypes.float32)
            combined_tensor_list = []
            expert_gate_list = []
            for i in range(self.top_k):
                expert_gate_list.append(expert_gate[:,i] * expert_mask_flat)
            for i in range(self.top_k):
                combined_tensor_list.append(tf.expand_dims(tf.expand_dims(expert_gate_list[i] * expert_mask_flat, axis=-1) * tf.one_hot(expert_index[:,i],depth=self.num_experts), axis=-1) * tf.one_hot(position_in_expert, depth=self.expert_capacity_int))
            combined_tensor_list_stacked = tf.stack(combined_tensor_list, axis=0)
            combined_tensor = tf.reduce_sum(combined_tensor_list_stacked, axis=0)
            dispatch_tensor = tf.cast(combined_tensor,tf.dtypes.bool)
            dispatch_tensor = tf.cast(dispatch_tensor, tf.dtypes.float32)
            # Mask out the experts that have overflowed the expert capacity.
            # Keep only tokens that fit within expert capacity.
            # Experts have a fixed capacity, ensure we do not exceed it. Construct
            # the batch indices, to each expert, with position in expert make sure that
            # not more that expert capacity examples can be routed to each expert.
       
        return dispatch_tensor, combined_tensor

    def get_config(self):
        config = {
            'num_experts': self.num_experts,
            'num_masked_experts': self.num_masked_experts,
            'expert_capacity': self.expert_capacity_int,
            'top_k': self.top_k,
            'aux_loss': self.aux_loss,
            'aux_loss_alpha': self.aux_loss_alpha,
            'num_tokens_per_sample': self.num_tokens_per_sample
        }
        
        return config
        

            

def VAE_encoder_mean(num_gauss=4):
    return tf.keras.layers.Dense(num_gauss, use_bias=False)
def VAE_encoder_std(num_gauss=16):
    return tf.keras.layers.Dense(num_gauss, activation='sigmoid', use_bias=False)

        


# Switch layer
class Switch(layers.Layer):
    def __init__(self, config, num_tokens_per_sample, num_experts,num_masked_experts, batch_size,
                 capacity_factor=1.0, switch_deepth=4,group_deepth=4,
                 top_k=1,aux_loss=True,aux_loss_alpha=0.01,aux_loss_beta=0.001):
        self.layer_num = config.transformer["num_layers"]
        self.num_experts = num_experts
        self.num_masked_experts = num_masked_experts
        self.batch_size = batch_size
        self.aux_loss_alpha = aux_loss_alpha
        self.aux_loss_beta = aux_loss_beta
        self.embed_dim = config.hidden_size
        self.switch_deepth = switch_deepth
        self.moe_mlp_dim = config.transformer["moe_mlp_dim"]
        hidden_units = [config.transformer["moe_mlp_dim"], config.hidden_size]
        #hidden_units = [config.transformer["mlp_dim"], config.hidden_size]
        self.dropout_rate = config.transformer["dropout_rate"]
        self.experts = [mlp(hidden_units, config.transformer["dropout_rate"]) for _ in range(self.num_experts)]
        #self.base_mlp = mlp(hidden_units, config.transformer["dropout_rate"])
        self.num_tokens_per_sample = num_tokens_per_sample
        self.num_tokens_per_batch = int(num_tokens_per_sample*batch_size)
        self.capacity_factor = capacity_factor
        self.expert_capacity = int(self.num_tokens_per_batch * capacity_factor * top_k// self.num_experts)
        self.top_k = top_k
        self.aux_loss = aux_loss
        self.router = Router(self.num_experts, self.num_masked_experts, self.expert_capacity, 
                             top_k=self.top_k, aux_loss=aux_loss, aux_loss_alpha=self.aux_loss_alpha,
                            num_tokens_per_sample = num_tokens_per_sample)
        #self.i_th_layer = 0
        self.dispatch_tensor = None
        self.combine_tensor = None
        self.group_deepth = group_deepth
        #encoder_mean = VAE_encoder_mean()
        #self.encoder_std = VAE_encoder_std()
        #self.fixed_capacity = fixed_capacity
        super(Switch, self).__init__()

    def call(self, inputs):
        
        #inputs_shape = inputs.get_shape()
        #batch_size = tf.shape(inputs)[0]
        #num_tokens_per_example = tf.shape(inputs)[1]
        #num_tokens_per_batch = self.batch_size*self.num_tokens_per_batch
        
        # inputs shape: [num_tokens_per_batch, embed_dim]
        
            
        inputs = tf.reshape(inputs, [self.num_tokens_per_batch, self.embed_dim])
        # dispatch_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        # combine_tensor shape: [tokens_per_batch, num_experts, expert_capacity]
        dispatch_tensor, combine_tensor = self.router(inputs)
        expert_inputs = tf.einsum("ab,acd->cdb", inputs, dispatch_tensor)

        expert_inputs = tf.reshape(
            expert_inputs, [self.num_experts, self.expert_capacity, self.embed_dim]
        )
        # Dispatch to experts
        expert_input_list = tf.unstack(expert_inputs, axis=0)
        expert_output_list = [
            self.experts[idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
        
        
        '''using_sliding_loss=False
        if using_sliding_loss==True:
            expert_output_list_sliding = []
            for idx, expert_input in enumerate(expert_input_list):
                if idx != len(expert_input_list)-1:
                    expert_output_list_sliding.append(self.experts[idx+1](expert_input))
                else:
                    expert_output_list_sliding.append(self.experts[0](expert_input))
            expert_output_1 = tf.stack(expert_output_list, axis=0)
            expert_output_1 = tf.reshape(expert_output_1,[-1, self.embed_dim])
            expert_output_2 = tf.stack(expert_output_list_sliding, axis=0)
            expert_output_2 = tf.reshape(expert_output_2,[-1, self.embed_dim])
            self.add_loss(sliding_loss(
                self.encoder_std(expert_output_1),self.encoder_std(expert_output_2),self.aux_loss_beta))'''
            
            
        # expert_outputs shape: [expert_capacity, num_experts, embed_dim]
        expert_outputs = tf.stack(expert_output_list, axis=1)
        #expert_outputs = tf.stack(expert_output_list, axis=0)
        # expert_outputs_combined shape: [tokens_per_batch, embed_dim]
        expert_outputs_combined = tf.einsum(
            "abc,xba->xc", expert_outputs, tf.cast(combine_tensor,tf.dtypes.float32)
        )
        # output shape: [batch_size, num_tokens_per_example, embed_dim]

        outputs = tf.reshape(
            expert_outputs_combined,
            [self.batch_size, self.num_tokens_per_sample, self.embed_dim]
        )
        return outputs

    def get_config(self):
        config = {
            'hidden_size': self.embed_dim,
            'num_layers': self.layer_num,
            'moe_mlp_dim': self.moe_mlp_dim,
            'dropout_rate': self.dropout_rate,
            'num_tokens_per_sample': self.num_tokens_per_sample,
            'num_experts': self.num_experts,
            'num_masked_experts': self.num_masked_experts,
            'batch_size': self.batch_size,
            'capacity_factor': self.capacity_factor,
            'switch_deepth': self.switch_deepth,
            'group_deepth': self.group_deepth,
            'top_k': self.top_k,
            'aux_loss': self.aux_loss,
            'aux_loss_alpha': self.aux_loss_alpha,
            'aux_loss_beta': self.aux_loss_beta
        }
        
        return config

    @classmethod
    def from_config(cls, config):
        config_dict = ml_collections.ConfigDict()
        config_dict.hidden_size = config['hidden_size']
        config_dict.transformer = ml_collections.ConfigDict()
        config_dict.transformer.num_layers = config['num_layers']
        config_dict.transformer.moe_mlp_dim = config['moe_mlp_dim']
        config_dict.transformer.dropout_rate = config['dropout_rate']
        
        return cls(
            config=config_dict, 
            num_tokens_per_sample=config['num_tokens_per_sample'], 
            num_experts=config['num_experts'], 
            num_masked_experts=config['num_masked_experts'], 
            batch_size=config['batch_size'], 
            capacity_factor=config['capacity_factor'], 
            switch_deepth=config['switch_deepth'], 
            group_deepth=config['group_deepth'], 
            top_k=config['top_k'], 
            aux_loss=config['aux_loss'], 
            aux_loss_alpha=config['aux_loss_alpha'], 
            aux_loss_beta=config['aux_loss_beta']
        )
        
        


