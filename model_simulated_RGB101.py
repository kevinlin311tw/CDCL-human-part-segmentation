
from keras.models import Model
from keras.layers.merge import Concatenate, Multiply
from keras.layers import Activation, Input, Lambda
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from keras import layers
from keras.layers import Dense, Flatten, Conv2D, UpSampling2D, Add, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.engine import Layer, InputSpec
from keras import initializers
from keras.layers import add

import mylayers
import code
import keras.backend as K

stages = 1
np_branch1 = 38
np_branch2 = 19
np_branch3 = 15


class Scale(Layer):
    """Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    Keyword arguments:
    axis -- integer, axis along which to normalize in mode 0. For instance,
        if your input tensor has shape (samples, channels, rows, cols),
        set axis to 1 to normalize per feature map (channels axis).
    momentum -- momentum in the computation of the exponential average 
        of the mean and standard deviation of the data, for 
        feature-wise normalization.
    weights -- Initialization weights.
        List of 2 Numpy arrays, with shapes:
        `[(input_shape,), (input_shape,)]`
    beta_init -- name of initialization function for shift parameter 
        (see [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.
    gamma_init -- name of initialization function for scale parameter (see
        [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.
        
    """
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def mytransform(source, ref_tensor):
    target_shape = K.shape(ref_tensor)
    source_shape = K.shape(source)
    return K.resize_images(source, target_shape[1]/source_shape[1],target_shape[2]/source_shape[2], "channels_last")
    # return tensorflow.image.resize_images(source, (target_shape[1],target_shape[2]))

def relu(x): return Activation('relu')(x)
def sigmoid(x): return Activation('sigmoid')(x)

def conv(x, nf, ks, name, weight_decay):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def conv_stride(x, nf, ks, name, weight_decay, stride=(2,2)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[1]) if weight_decay else None

    x = Conv2D(nf, (ks, ks), padding='same', name=name, strides=stride,
               kernel_regularizer=kernel_reg,
               bias_regularizer=bias_reg,
               kernel_initializer=random_normal(stddev=0.01),
               bias_initializer=constant(0.0))(x)
    return x

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity_block is the block that has no conv layer at shortcut
    
    Keyword arguments
    input_tensor -- input tensor
    kernel_size -- defualt 3, the kernel size of middle conv layer at main path
    filters -- list of integers, the nb_filters of 3 conv layer at main path
    stage -- integer, current stage label, used for generating layer names
    block -- 'a','b'..., current block label, used for generating layer names
    
    """
    eps = 1.1e-5
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    
    Keyword arguments:
    input_tensor -- input tensor
    kernel_size -- defualt 3, the kernel size of middle conv layer at main path
    filters -- list of integers, the nb_filters of 3 conv layer at main path
    stage -- integer, current stage label, used for generating layer names
    block -- 'a','b'..., current block label, used for generating layer names
        
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    
    """
    eps = 1.1e-5
    
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size),name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1),name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def ResNet101_graph(img_input, weight_decay):
    eps = 1.1e-5

    branch = 0
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # C1 --------------------------------------------------
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    C1 = x

    # C2 --------------------------------------------------
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    C2 = x

    # C3 --------------------------------------------------
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 3):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))
    C3 = x

    # C4 --------------------------------------------------
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))
    C4 = x

    # C5 ---------------------------------------------------
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    C5 = x

    return C1, C2, C3, C4, C5


def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P1 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced')(C1)

    # upsample P5 to get P5_up1
    P5_up1 = mylayers.DeformableDeConv(name='P5_up1_deconv',
                                             kernel_size=4,
                                             stride=[1,2,2,1],
                                             filter_num=feature_size)([P5,P4])
    # upsample P5_up1 to get P5_up2
    P5_up2 = mylayers.DeformableDeConv(name='P5_up2_deconv',
                                             kernel_size=4,
                                             stride=[1,2,2,1],
                                             filter_num=feature_size)([P5_up1,P3])
    # upsample P4 to get P4_up1
    P4_up1 = mylayers.DeformableDeConv(name='P4_up1_deconv',
                                             kernel_size=4,
                                             stride=[1,2,2,1],
                                             filter_num=feature_size)([P4,P3])

    # downsample P1 to get P1_down1
    P1_down1 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P1_down1')(P1)
    # downsample P1_down1 to get P1_down2
    P1_down2 = Conv2D(feature_size, kernel_size=1, strides=2, padding='same', name='P1_down2')(P1_down1)
    # downsample P2 to get P2_down
    P2_down1 = Conv2D(feature_size, kernel_size=1, strides=2, padding='same', name='P2_down1')(P2)


    P5_up2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5_up2_head')(P5_up2)
    P5_up2 = relu(P5_up2)

    P4_up1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4_up1_head')(P4_up1)
    P4_up1 = relu(P4_up1)

    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3_head')(P3)
    P3 = relu(P3)

    P2_down1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2_down1_head')(P2_down1)
    P2_down1 = relu(P2_down1)

    P1_down2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1_down2_head')(P1_down2)
    P1_down2 = relu(P1_down2)


    # Concatenate features at different levels
    pyramid_feat = []
    pyramid_feat.append(P5_up2)
    pyramid_feat.append(P4_up1)
    pyramid_feat.append(P3)
    pyramid_feat.append(P2_down1)
    pyramid_feat.append(P1_down2)
    feats = Concatenate()(pyramid_feat)

    return feats



def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 512, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "Mconv6_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 1, "Mconv7_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv8_stage1_L%d" % branch, (weight_decay, 0))

    return x



def stage1_segmentation_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 256, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv6_stage1_L%d" % branch, (weight_decay, 0))
    #x = sigmoid(x)
    x = Activation('softmax')(x)

    return x

def apply_mask(x, mask1, mask2, mask3, num_p, stage, branch):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if num_p == np_branch1:
        w = Multiply(name=w_name)([x, mask1])  # vec_weight
    elif num_p == np_branch2:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    elif num_p == np_branch3:
        w = Multiply(name=w_name)([x, mask3])  # seg
    else:
        assert False, "wrong number of layers num_p=%d " % num_p
    return w


def get_training_model_resnet101(weight_decay, gpus=None):

    img_input_shape = (None, None, 3)
    vec_input_shape = (None, None, 38)
    heat_input_shape = (None, None, 19)
    seg_input_shape = (None, None, 15)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)
    seg_weight_input = Input(shape=seg_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)
    inputs.append(seg_weight_input)

    # resnet101
    C1, C2, C3, C4, C5 = ResNet101_graph(img_input, weight_decay)

    stage0_out = create_pyramid_features(C1, C2, C3, C4, C5)

    # Additional layers for learning multi-scale semantics
    stage0_out = conv(stage0_out, 512, 3, "pyramid_1_CPM", (weight_decay, 0))
    stage0_out = relu(stage0_out)
    stage0_out = conv(stage0_out, 512, 3, "pyramid_2_CPM", (weight_decay, 0))
    stage0_out = relu(stage0_out)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, seg_weight_input, np_branch1, 1, 1)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, seg_weight_input, np_branch2, 1, 2)

    # stage 1 - branch 3 (semantic segmentation)
    stage1_branch3_out = stage1_segmentation_block(stage0_out, np_branch3, 3, weight_decay)
    w3 = apply_mask(stage1_branch3_out, vec_weight_input, heat_weight_input, seg_weight_input, np_branch3, 1, 3)

    outputs.append(w1)
    outputs.append(w2)
    outputs.append(w3)


    if gpus is None:
        model = Model(inputs=inputs, outputs=outputs)
    else:
        import tensorflow as tf
        with tf.device('/cpu:0'): #this model will not be actually used, it's template
            model = Model(inputs=inputs, outputs=outputs)

    return model



def get_testing_model_resnet101():

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    C1, C2, C3, C4, C5 = ResNet101_graph(img_input, None)

    stage0_out = create_pyramid_features(C1, C2, C3, C4, C5)

    # Additional layers for learning multi-scale semantics
    stage0_out = conv(stage0_out, 512, 3, "pyramid_1_CPM", (None, 0))
    stage0_out = relu(stage0_out)
    stage0_out = conv(stage0_out, 512, 3, "pyramid_2_CPM", (None, 0))
    stage0_out = relu(stage0_out)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    # stage 1 - branch 3 (semantic segmentation)
    stage1_branch3_out = stage1_segmentation_block(stage0_out, np_branch3, 3, None)


    model = Model(inputs=[img_input], outputs=[stage1_branch1_out, stage1_branch2_out, stage1_branch3_out])
    return model

