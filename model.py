#%%
import tensorflow as tf
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras import activations
from group_convolution import GroupConv2D



class BottleNeck(tf.keras.Model):
    def __init__(self, in_channel, mid_channel, out_channel, strides, radix, groups, reduction_factor):
        super(BottleNeck, self).__init__()
        self.radix = radix
        self.groups = groups
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.inter_channels = max(in_channel*radix//reduction_factor, 32)
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv1 = GroupConv2D(input_channels=in_channel,
                                output_channels=mid_channel,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                groups=groups*radix,
                                use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv2 = GroupConv2D(input_channels=mid_channel,
                                output_channels=mid_channel*radix,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                groups=groups*radix,
                                use_bias=False)
        self.fc1 = GroupConv2D(input_channels=self.mid_channel,
                                output_channels=self.inter_channels,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                groups=self.groups,
                                use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.fc2 = GroupConv2D(input_channels=self.inter_channels,
                                output_channels=self.mid_channel*self.radix,
                                kernel_size=1,
                                strides=1,
                                padding='same',
                                groups=self.groups,
                                use_bias=False)                       
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.bn4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.conv4 = tf.keras.layers.Conv2D(self.out_channel,kernel_size=1,strides=strides,padding='same',use_bias=False)
        if self.in_channel != self.out_channel:
            self.conv3 = tf.keras.layers.Conv2D(self.out_channel,kernel_size=1,strides=1,padding='same',use_bias=False)
        else:
            self.sub_sample = tf.keras.layers.MaxPooling2D(pool_size=2,strides=strides,padding='same')
    def call(self, inputs, training=None, **kwargs):

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        splited = tf.split(x, num_or_size_splits=self.radix, axis=-1)
        z = tf.reshape(self.global_pool(sum(splited)),[-1,1,1,self.mid_channel])
        z = self.fc1(z)
        z = self.bn3(z, training=training)
        z = tf.nn.relu(z)
        z = self.fc2(z)
        #实现r-softmax
        z = tf.reshape(z,[-1,self.groups,self.radix,self.mid_channel//self.groups])
        z = tf.transpose(z,[0,2,1,3])
        z = tf.reshape(z,[-1,self.radix,self.mid_channel])
        z = tf.keras.activations.softmax(z,axis=1)
        
        logits = [tf.expand_dims(m,axis=1) for m in tf.split(z,num_or_size_splits=self.radix,axis=1)]
        out = sum([a*b for a,b in zip(splited,logits)])
        out = tf.nn.relu(self.bn4(out))
        out = self.conv4(out)
        if self.in_channel!= self.out_channel:
            shortcut = self.conv3(inputs)
        else:
            shortcut = self.sub_sample(inputs)

        output = out + shortcut

        return output
            




        return x



class ResNestBlock(tf.keras.Model):
    def __init__(self, num_layers, in_channel, mid_channel, out_channel, strides, radix, groups, reduction_factor):
        super(ResNestBlock,self).__init__()
        self.radix = radix
        self.groups = groups
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.num_layers = num_layers
        self.listLayers = []
        for _ in range(self.num_layers):
            self.listLayers.append(BottleNeck(in_channel=self.in_channel,
                                            mid_channel=self.mid_channel,
                                            out_channel=self.out_channel,
                                            strides=strides,
                                            radix=self.radix,
                                            groups=self.groups,
                                            reduction_factor=reduction_factor))

    def call(self, x, training=None, **kwargs):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionLayer(tf.keras.Model):
    def __init__(self, in_channel, mid_channel, out_channel, strides, radix, groups, reduction_factor):
        super(TransitionLayer, self).__init__()
        self.radix = radix
        self.groups = groups
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        self.tran = BottleNeck(in_channel=self.in_channel,
                                mid_channel=self.mid_channel,
                                out_channel=self.out_channel,
                                strides=strides,
                                radix=self.radix,
                                groups=self.groups,
                                reduction_factor=reduction_factor)
    def call(self, inputs, training=None, **kwargs):
        x = self.tran(inputs)
        return x


class ResNest(tf.keras.Model):
    def __init__(self, num_class, num_init_features, block_layers, layers_num, radix, groups, reduction_factor):
        super(ResNest, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=num_init_features,
                                           kernel_size=(7, 7),
                                           strides=2,
                                           use_bias=False,
                                           padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                              strides=2,
                                              padding="same")
        self.transition_11 = TransitionLayer(in_channel=num_init_features,
                                            mid_channel=layers_num[0]//4,
                                            out_channel=layers_num[0],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)                                              
        self.resnetst_block_1 = ResNestBlock(num_layers=block_layers[0]-1,
                                            in_channel = layers_num[0],
                                            mid_channel=layers_num[0]//4,
                                            out_channel=layers_num[0],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)
        self.transition_12 = TransitionLayer(in_channel=layers_num[0],
                                            mid_channel=layers_num[0]//4,
                                            out_channel=layers_num[0],
                                            strides=2,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)
        self.transition_21 = TransitionLayer(in_channel=layers_num[0],
                                            mid_channel=layers_num[1]//4,
                                            out_channel=layers_num[1],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)                                              
        self.resnetst_block_2 = ResNestBlock(num_layers=block_layers[1]-1,
                                            in_channel = layers_num[1],
                                            mid_channel=layers_num[1]//4,
                                            out_channel=layers_num[1],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)
        self.transition_22 = TransitionLayer(in_channel=layers_num[1],
                                            mid_channel=layers_num[1]//4,
                                            out_channel=layers_num[1],
                                            strides=2,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)                                                 
        
        self.transition_31 = TransitionLayer(in_channel=layers_num[1],
                                            mid_channel=layers_num[2]//4,
                                            out_channel=layers_num[2],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)                                              
        self.resnetst_block_3 = ResNestBlock(num_layers=block_layers[2]-1,
                                            in_channel = layers_num[2],
                                            mid_channel=layers_num[2]//4,
                                            out_channel=layers_num[2],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)
        self.transition_32 = TransitionLayer(in_channel=layers_num[2],
                                            mid_channel=layers_num[2]//4,
                                            out_channel=layers_num[2],
                                            strides=2,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)

        self.transition_41 = TransitionLayer(in_channel=layers_num[2],
                                            mid_channel=layers_num[3]//4,
                                            out_channel=layers_num[3],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)                                              
        self.resnetst_block_4 = ResNestBlock(num_layers=block_layers[3]-1,
                                            in_channel = layers_num[3],
                                            mid_channel=layers_num[3]//4,
                                            out_channel=layers_num[3],
                                            strides=1,
                                            radix=radix,
                                            groups=groups,
                                            reduction_factor=reduction_factor)


        self.bn2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_class,
                                        activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)
        x = self.transition_11(x, training=training)
        x = self.resnetst_block_1(x, training=training)
        x = self.transition_12(x, training=training)

        x = self.transition_21(x, training=training)
        x = self.resnetst_block_2(x, training=training)
        x = self.transition_22(x, training=training)

        x = self.transition_31(x, training=training)
        x = self.resnetst_block_3(x, training=training)
        x = self.transition_32(x, training=training)

        x = self.transition_41(x, training=training)
        x = self.resnetst_block_4(x, training=training)



        x= self.bn2(x)
        x = tf.nn.relu(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x




def ResNetst50(num_class):
    return ResNest(num_class=num_class,num_init_features=64,block_layers=[2,3,5,3],layers_num=[256,512,1024,2048],
    radix=4, groups=2, reduction_factor=8)

                    
def build(num_class,length,channel):
    model = ResNetst50(num_class)
    return model


# %%
#model = ResNetst50(12)
#model.build(input_shape=(None,128,2))

#model.summary()

# # %%
# data = tf.random.normal([12,128,128,2])
# label = tf.constant([2,1,3,4,5,6,4,3,5,6,7,3])
# label = tf.one_hot(label,depth=NUM_CLASSES)

# print(data.shape,label.shape)

# # %%
# optimizers = tf.keras.optimizers.Adam()
# model.compile(optimizer=optimizers, loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(data,label,batch_size=1)

# %%

