import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from Utils import load_all_classes,sep_triplet_loss
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
images1Dir = 'D:\\DataSets\\CASIA-B\\001\\001\\nm-01\\090'
images2Dir = 'D:\\DataSets\\CASIA-B\\002\\002\\nm-01\\090'
NUMCLASSES=10
CLIPSIZE=3
NUM_B3DABLOCK = 6

def initVariable(initializer):
    weightLayer1 = tf.Variable(initializer(shape=[3,3,3,1,1],dtype=tf.float32))
    weightList = [weightLayer1]
    biasList = tf.Variable(initializer(shape=[NUM_B3DABLOCK*4 + 1],dtype=tf.float32))
    for i in range(NUM_B3DABLOCK):
        weightBlock = []
        for shape in [[3,3,3,1,8],[1,3,3,8,4],[3,1,1,4,2],[1,1,1,2,1]]:
            weight = tf.Variable(initializer(shape=shape,dtype=tf.float32))
            weightBlock.append(weight)
        weightList.append(weightBlock)
    return weightList,biasList



def Myconv3D(tensor,filter,bias):
    result = tf.nn.conv3d(tensor,filter,strides=[1, 1, 1, 1, 1],padding='SAME')
    result = result + bias
    return result

def localTransform(imgs,s):
    print('localTransfrom shape',imgs.shape)
    k = imgs.shape[1]//s
    imgs = imgs[:,:k*s]
    #imgs.shape = (54,102,102)
    imgs = tf.split(imgs,num_or_size_splits=k,axis=1)
    clip = tf.convert_to_tensor(imgs)
    print(clip.shape,'aaa')
    clip = clip[:,:,:,:,:,0]
    print(clip.shape, 'aaa')
    #(18,3,102,102)
    # clip = tf.expand_dims(clip,0)
    clip = tf.transpose(clip, perm=[1, 0, 2, 3, 4])
    clip = tf.transpose(clip,perm=[0,1,4,3,2])
    clip = tf.transpose(clip, perm=[0, 1, 3, 2, 4])
    clip = tf.convert_to_tensor(clip,tf.float32)
    #inputShape = (1, 18, 102, 102, 3)
    filterShape = np.array([3,1,1,3,1])
    filter = tf.ones(shape=filterShape)
    clip = tf.nn.conv3d(clip,filter,strides=[1, 1, 1, 1, 1], padding='SAME')
    print(clip.shape)
    return clip

def framePooling(batch,axi):
    mean = tf.reduce_mean(batch,axis=axi)
    max = tf.reduce_max(batch,axis=axi)
    return mean + max

class B3D_A(tf.keras.Model):
    def __init__(self,filters,bias):
        super(B3D_A, self).__init__(name='B3D_A')
        self.filters1,self.filters2,self.filters3,self.filters4 = filters
        self.bias1 = bias[0]
        self.bias2 = bias[1]
        self.bias3 = bias[2]
        self.bias4 = bias[3]
    def call(self, input_tensor, training=False, mask=None):
        x1 = Myconv3D(input_tensor,self.filters1,self.bias1)
        x2 = Myconv3D(x1,self.filters2,self.bias2)
        x2 = Myconv3D(x2,self.filters3,self.bias3)
        x2 = Myconv3D(x2,self.filters4,self.bias4)
        x = x1 + x2
        return tf.nn.relu(x)

class B3D_B(tf.keras.Model):
    def __init__(self,filters):
        super(B3D_B, self).__init__(name='B3D_B')
        filters1,filters2,filters3 = filters

        self.conv3a = tf.keras.layers.Conv3D(filters1,[1,3,3,3,1])
        self.conv3b = tf.keras.layers.Conv3D(filters2,[1,1,3,3,1])
        self.conv3c = tf.keras.layers.Conv3D(filters3,[1,3,1,1,1])

    def call(self, input_tensor, training=False, mask=None):
        x1 = self.conv3a(input_tensor)
        x1 = tf.nn.relu(x1)
        x2 = self.conv3b(input_tensor)
        x2 = tf.nn.relu(x2)
        x3 = self.conv3c(input_tensor)
        x3 = tf.nn.relu(x3)
        x = x1 + x2 + x3
        return tf.nn.relu(x)

class Layer1(tf.keras.layers.Layer):
    def __init__(self,num_outputs,filters,bias):
        super(Layer1, self).__init__()
        self.num_outputs = num_outputs
        self.filters = filters
        self.bias = bias

    def call(self, inputs, **kwargs):
        print('layer1 input.shape',inputs.shape)
        # inputs = inputs[0, :, :, :]
        # inputs = tf.expand_dims(inputs,0)
        inputs = tf.expand_dims(inputs,4)
        smScaleX = Myconv3D(inputs,filter=self.filters[0], bias=self.bias[0])
        lgScaleX = localTransform(smScaleX,CLIPSIZE)
        return smScaleX,lgScaleX

class Layer2(tf.keras.layers.Layer):
    def __init__(self,num_outputs,b3DA1,b3DA2):
        super(Layer2, self).__init__()
        self.num_outputs = num_outputs
        self.b3DA1 = b3DA1
        self.b3DA2 = b3DA2
    def call(self, inputs, **kwargs):
        smScaleX,lgScaleX = inputs
        smScaleX = self.b3DA1(smScaleX)
        smScaleX = tf.nn.max_pool3d(smScaleX,(1,2,2),1,'SAME')
        lgScaleX = self.b3DA2(lgScaleX)
        lgScaleX = tf.nn.max_pool3d(lgScaleX,(1,2,2),1,'SAME')
        localT = localTransform(smScaleX,CLIPSIZE)
        lgScaleX += localT
        return smScaleX,lgScaleX

class Layer3(tf.keras.layers.Layer):
    def __init__(self,num_outputs,b3DA3,b3DA4,b3DA5,b3DA6):
        super(Layer3, self).__init__()
        self.num_outputs = num_outputs
        self.b3DA3 = b3DA3
        self.b3DA4 = b3DA4
        self.b3DA5 = b3DA5
        self.b3DA6 = b3DA6

    def call(self, inputs, **kwargs):
        smScaleX, lgScaleX = inputs
        smScaleX = self.b3DA3(smScaleX)
        smScaleX = self.b3DA4(smScaleX)
        lgScaleX = self.b3DA5(lgScaleX)
        lgScaleX = self.b3DA6(lgScaleX)
        localT = localTransform(smScaleX,CLIPSIZE)
        lgScaleX += localT
        return smScaleX,lgScaleX




class MT3DCNN(tf.keras.Model):
    def __init__(self,filters,bias):
        super(MT3DCNN, self).__init__()
        self.filters = filters
        self.bias = bias
        self.layer1 = Layer1(1, filters=self.filters, bias=self.bias)
        self.b3DA1 = B3D_A(self.filters[1],self.bias[1:5])
        self.b3DA2 = B3D_A(self.filters[2], self.bias[5:9])
        self.layer2 = Layer2(2,self.b3DA1,self.b3DA2)
        self.b3DA3 = B3D_A(self.filters[3],self.bias[9:13])
        self.b3DA4 = B3D_A(self.filters[4], self.bias[13:17])
        self.b3DA5 = B3D_A(self.filters[5], self.bias[17:21])
        self.b3DA6 = B3D_A(self.filters[6], self.bias[21:25])
        self.layer3 = Layer3(2, self.b3DA3,self.b3DA4,self.b3DA5,self.b3DA6)
        self.flatten1 = tf.keras.layers.Flatten(input_shape=(1,44,64))
        self.flatten2 = tf.keras.layers.Flatten(input_shape=(1, 44, 64))
        # self.bathNormalization1 = tf.keras.layers.BatchNormalization()
        # self.bathNormalization2 = tf.keras.layers.BatchNormalization()
        # self.bathNormalization3 = tf.keras.layers.BatchNormalization()
        # self.bathNormalization4 = tf.keras.layers.BatchNormalization()
        self.bathNormalization5 = tf.keras.layers.BatchNormalization()
        self.bathNormalization6 = tf.keras.layers.BatchNormalization()
        self.bathNormalization7 = tf.keras.layers.BatchNormalization()
        self.bathNormalization8 = tf.keras.layers.BatchNormalization()
        # self.dense1 = tf.keras.layers.Dense(units=22, activation=tf.nn.relu,use_bias=True,trainable=True)
        # self.dense2 = tf.keras.layers.Dense(units=22, activation=tf.nn.relu, use_bias=True, trainable=True)
        # self.dense3 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, use_bias=True, trainable=True)
        # self.dense4 = tf.keras.layers.Dense(units=16, activation=tf.nn.relu, use_bias=True, trainable=True)
        self.dense5 = tf.keras.layers.Dense(units=2, activation=tf.nn.relu, use_bias=True, trainable=True)
        self.dense6 = tf.keras.layers.Dense(units=2, activation=tf.nn.relu, use_bias=True, trainable=True)
        self.dense7 = tf.keras.layers.Dense(units=2, activation=tf.nn.relu, use_bias=True, trainable=True)
        self.dense8 = tf.keras.layers.Dense(units=2, activation=tf.nn.relu, use_bias=True, trainable=True)
        self.dense9 = tf.keras.layers.Dense(units=NUMCLASSES, activation=None, use_bias=True, trainable=True)
        self.dense10 = tf.keras.layers.Dense(units=NUMCLASSES, activation=None, use_bias=True, trainable=True)
        # self.pool1 = tf.keras.layers.MaxPool1D(pool_size=22,strides=None,padding='VALID',data_format='channels_last')
        # self.pool2 = tf.keras.layers.MaxPool1D(pool_size=22, strides=None, padding='VALID', data_format='channels_last')
        self.lamda1 = tf.keras.layers.Lambda(lambda x:tf.math.l2_normalize(x, axis=1))
        self.lamda2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))


    def call(self, inputs, training=None, mask=None):

        # inputs = inputs[0,:,:,:,:]
        print('inputs shape',inputs.shape)
        smScaleX, lgScaleX = self.layer1(inputs)
        print('Bf ly2', smScaleX.shape)
        print('Bf ly2', lgScaleX.shape)
        smScaleX, lgScaleX = self.layer2((smScaleX, lgScaleX))
        print('Bf ly3', smScaleX.shape)
        print('Bf ly3', lgScaleX.shape)
        smScaleX, lgScaleX = self.layer3((smScaleX, lgScaleX))
        smScaleX = framePooling(smScaleX,axi=1)
        lgScaleX = framePooling(lgScaleX,axi=1)
        smScaleX = framePooling(smScaleX,axi=3)
        lgScaleX = framePooling(lgScaleX,axi=3)
        smScaleX = self.flatten1(smScaleX)
        lgScaleX = self.flatten2(lgScaleX)
        print('Bf dence',smScaleX.shape)
        print('Bf dence', lgScaleX.shape)
        x = self.dense5(smScaleX)
        x = self.bathNormalization5(x)

        y = self.dense6(lgScaleX)
        y = self.bathNormalization6(y)

        x = self.dense7(x)
        x = self.bathNormalization7(x)

        y = self.dense8(y)
        y = self.bathNormalization8(y)

        x = self.dense9(x)

        y = self.dense10(y)

        # sum = tf.concat([x,y],axis=1)

        x = self.lamda1(x)
        y = self.lamda2(y)

        # outputs = tf.stack([x,y],axis=0)
        outputs = x+y

        print('outputs.shape:',outputs.shape)
        # outputs = tf.transpose(outputs)
        return outputs


path = 'D:\\DataSets\\CASIA-B\\001\\001\\nm-01\\090\\'
# print(tfa.losses.triplet_semihard_loss([0, 0], [[0.0, 1.0], [1.0, 0.0]]))
filters, bias = initVariable(tf.random_normal_initializer(mean=1., stddev=2.))
model = MT3DCNN(filters,bias)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss=tfa.losses.TripletSemiHardLoss(margin=0.2),
              # loss=sep_triplet_loss,
              metrics=['accuracy'])

# imgTensor1 = getImgs(images1Dir)
# imgTensor1 = getImgs(images1Dir)[:56,:,:]
# imgTensor2 = getImgs(images2Dir)[:56,:,:]
# imgTensor2 = getImgs(images2Dir)[:56,:,:]
# imgTensor = tf.stack([imgTensor1,imgTensor2],axis=0)
imgTensor,labels = load_all_classes(NUMCLASSES)
count = imgTensor.shape[0]-imgTensor.shape[0]%NUMCLASSES
imgTensor = imgTensor[:count,:,:,:]
labels = labels[:count]
print('imgTensor.shape',imgTensor.shape)
train_dataset = tf.data.Dataset.from_tensors((imgTensor,labels))
train_dataset.shuffle(1024)
train_dataset.batch(32)
train_dataset.repeat(200)

print('train_dataset:',train_dataset)
print('labels.len:',len(labels))
# inputs = tf.keras.Input(shape=dataset.shape)
# tf.config.experimental_run_functions_eagerly(True)
# model.build(imgTensor.shape)
# model.summary()
history = model.fit(train_dataset,
                    batch_size=16,
                    epochs=5)