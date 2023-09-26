import numpy as np
import tensorflow as tf
import os
import cv2
from PIL import Image
import tensorflow_addons as tfa

def cut(image):
    image = np.array(image)

    # 找到人的最小最大高度与宽度
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    # 设置切割后图片的大小，为size*size，因为人的高一般都会大于宽
    size = height_max - height_min
    temp = np.zeros((size, size))

    # 将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    # l = (width_max-width_min)//2
    # r = width_max-width_min-l
    # 以头为中心，将将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    l1 = head_top - width_min
    r1 = width_max - head_top
    # 若宽大于高，或头的左侧或右侧身子比要生成图片的一般要大。则此图片为不符合要求的图片
    flag = False
    if size <= width_max - width_min or size // 2 < r1 or size // 2 < l1:
        flag = True
        return temp, flag
    # centroid = np.array([(width_max+width_min)/2,(height_max+height_min)/2],dtype='int')
    temp[:, (size // 2 - l1):(size // 2 + r1)] = image[height_min:height_max, width_min:width_max]

    return temp, flag



def load_preprosess_image(input_path):
    image = tf.io.read_file(input_path) # 读取的是二进制格式 需要进行解码
    image = tf.image.decode_jpeg(image,channels=3)  # 解码 是通道数为3
    image = tf.image.resize(image,[256,256]) # 统一图片大小
    image = tf.cast(image,tf.float32) # 转换类型
    image = image/255 # 归一化
    return image

def getImgs(path):
    imgTensor = None
    for root,dirs,files in os.walk(path):
        for file in files:
            absPath = os.path.join(root,file)
            img = Image.open(absPath)
            cutImg,flag = cut(img)
            # print(cutImg.shape)
            # cutImg = cv2.resize(cutImg,(64,44))
            cutImg = cv2.resize(cutImg, (64, 44))
            img = tf.convert_to_tensor(cutImg)
            img = tf.expand_dims(img,0)
            img = tf.expand_dims(img,0)
            if imgTensor is None:
                imgTensor = img
            else:
                imgTensor = tf.concat([imgTensor,img],axis=1)
    return imgTensor

def cutRound(tensor):
    tensor = tensor[:,:40,:,:]
    return tensor
def load_one_class(name,className):
    imgTensor = None
    for i in range(1,7):
        path = name + '\\nm-0' + str(i) + '\\090\\'
        img = getImgs(path)
        if img.shape[1]<40:
            continue
        if imgTensor is None:
            imgTensor = cutRound(img)
        else:
            imgTensor = tf.concat([imgTensor,cutRound(img)],axis=0)
    labels = tf.fill([imgTensor.shape[0]],className)
    return imgTensor,labels

def load_all_classes(class_num):
    data,labels = load_one_class('D:\\DataSets\\CASIA-B\\001\\001',className=1)
    for i in range(2,class_num+1):
        num = ''
        if i < 10:
            num = '00' + str(i)
        elif i < 100:
            num = '0' + str(i)
        else:
            num = str(i)
        dataTemp,labelsTemp = load_one_class('D:\\DataSets\\CASIA-B\\{0}\\{0}'.format(num,num),className=i)
        data = tf.concat([dataTemp,data],axis=0)
        labels = tf.concat([labelsTemp, labels], axis=0)
    return data,labels

def _pairwise_distances(embeddings, squared=False):
    """
    计算嵌入向量之间的距离
    Args:
        embeddings: 形如(batch_size, embed_dim)的张量
        squared: Boolean. True->欧式距离的平方，False->欧氏距离
    Returns:
        piarwise_distances: 形如(batch_size, batch_size)的张量
    """
    # 嵌入向量点乘，输出shape=(batch_size, batch_size)
    dot_product = tf.matmul(embeddings, tf.transpose(embeddings))

    # 取dot_product对角线上的值，相当于是每个嵌入向量的L2正则化，shape=(batch_size,)
    square_norm = tf.compat.v1.diag_part(dot_product)

    # 计算距离,shape=(batch_size, batch_size)
    # ||a - b||^2 = ||a||^2 - 2<a, b> + ||b||^2
    # PS: 下面代码计算的是||a - b||^2，结果是一样的
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)

    # 保证距离都>=0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # 加一个接近0的值，防止求导出现梯度爆炸的情况
        mask = tf.compat.v1.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # 校正距离
        distances = distances * (1.0 - mask)
    return distances

def _get_triplet_mask(labels):
    """
    Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # i, j, k分别是不同的样本索引
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    # combine the two masks
    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask

def batch_all_triplet_loss(labels, embeddings, margin, squared=False):
    """
    计算整个banch的Triplet loss。
    生成所有合格的triplets样本组，并只对其中>0的部分取均值
    Args:
        labels: 标签，shape=(batch_size,)
        embeddings: 形如(batch_size, embed_dim)的张量
        margin: Triplet loss中的间隔
        squared: Boolean. True->欧氏距离的平方，False->欧氏距离
    Returns:
        triplet_loss: 损失
    """
    # 获取banch中嵌入向量间的距离矩阵
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    anchor_positive_dist = tf.expand_dims(pairwise_dist, 2)
    anchor_negative_dist = tf.expand_dims(pairwise_dist, 1)

    # 计算一个形如(batch_size, batch_size, batch_size)的3D张量
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    # 将invalid Triplet置零
    # label(a) != label(p) or label(a) == label(n) or a == p
    mask = _get_triplet_mask(labels)
    mask = tf.compat.v1.to_float(mask)
    triplet_loss = tf.multiply(mask, triplet_loss)

    # 删除负值
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    # 计算正值
    valid_triplets = tf.compat.v1.to_float(tf.greater(triplet_loss, 1e-16))
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
    return triplet_loss, fraction_positive_triplets

def _get_anchor_positive_triplet_mask(labels):
    """
    返回一个2D掩码，掩码用于筛选合格的同类样本对[a, p]。合格的要求是：a和p是不同的样本索引，a和p具有相同的标签。
    Args:
        labels: tf.int32 形如[batch_size]的张量
    Returns:
        mask: tf.bool 形如[batch_size]的张量
    """
    # i和j是不同的
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    # label[i] == label[j]
    labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    # 合并
    mask = tf.logical_and(indices_not_equal, labels_equal)
    return mask

def batch_hard_triplet_loss(labels, embeddings, margin, squared=False):
    """
    为该batch计算Triplet loss
    遍历所有样本，将其作为原点anchor，获取hardest同类和一类样本，构建一个Triplet
    """
    # 获得一个2D的距离矩阵，表示嵌入向量之间的欧氏距离
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # 合格的同类样本距离
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = tf.compat.v1.to_float(mask_anchor_positive)
    anchor_positive_dist = tf.multiply(mask_anchor_positive, pairwise_dist)

    # 对每行取最大距离，每行表示每个anchor，输出shape=(batch_size, 1)
    hardest_positive_dist = tf.reduce_max(anchor_positive_dist, axis=1, keepdims=True)

    # 合格的异类样本距离矩阵的掩码
    mask_anchor_negative = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_negative = tf.compat.v1.to_float(mask_anchor_negative)

    # 获取每个anchor下的嵌入向量样本对的最大距离
    max_anchor_negative_dist = tf.reduce_max(pairwise_dist, axis=1, keepdims=True)
    # 不合格的negative嵌入向量样本对距离都要在原来的基础上 + 上面的max_anchor_negative_dist
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
    # 在每行选择最小距离
    hardest_negative_dist = tf.reduce_min(anchor_negative_dist, axis=1, keepdims=True)

    triplet_loss = tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0)

    triplet_loss = tf.reduce_mean(triplet_loss)
    return triplet_loss
def sep_triplet_loss(y_true, y_pred):
    print('running loss func')
    print(y_pred.shape)
    y_pred1 = y_pred[0]
    y_pred2 = y_pred[1]
    y_true = y_true[:,0]
    print('y_pred1',y_pred1)
    print('y_pred2',y_pred2)
    print('y_true', y_true)
    # loss1 = tfa.losses.triplet_semihard_loss(y_true,y_pred1,margin=0.2)
    # loss2 = tfa.losses.triplet_semihard_loss(y_true,y_pred2,margin=0.2)
    loss1 = batch_hard_triplet_loss(y_true,y_pred1,0.2)
    print('loss1',loss1)
    loss2 = batch_hard_triplet_loss(y_true,y_pred2,0.2)
    print('loss2:', loss2)
    loss = loss1 + loss2
    print('loss:',loss)
    return loss





