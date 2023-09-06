import torch
import numpy as np
from scipy.ndimage import zoom
# import maxflow
# import maxflow_dgcn_cpp


def dgcn_cacul_knn_matrix_(feature_map, k=10):
    batchsize, channels, h, w = feature_map.shape
    n = h * w
    # S = torch.zeros(batchsize,n,n)
    knn_matrix = torch.zeros(batchsize, n, n, device='cuda')
    for i in range(batchsize):
        # reshape feature_map: n*channel
        feature = torch.transpose(feature_map[i].reshape(channels, h * w), 0, 1)
        # ||x1-x2||^2
        x1_norm = (feature ** 2).sum(dim=1).view(-1, 1)  # n*1
        x2_norm = x1_norm.view(1, -1)  # 1*n
        dist = (x1_norm + x2_norm - 2.0 * torch.mm(feature, feature.transpose(0, 1))).abs()  # x1_norm + x2_norm : n*n
        # first method
        value, position = torch.topk(dist, k, dim=1, largest=False)  # value's shape[n, 10]
        temp = value[:, -1].unsqueeze(1).repeat(1, n)

        knn_matrix[i] = (dist <= temp).float() - torch.eye(n, n, device='cuda')

    return knn_matrix


def dgcn_crf_operation(images, probs, img_metas):
    img_mean_4_dataset = img_metas[0]['img_norm_cfg']['mean']
    img_std_4_dataset = img_metas[0]['img_norm_cfg']['std']

    batchsize, _, h, w = probs.shape
    probs[probs < 0.0001] = 0.0001
    # unary = np.transpose(probs, [0, 2, 3, 1])

    im = images
    im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
    im = np.transpose(im, [0, 2, 3, 1])
    im = im * img_std_4_dataset[None, None, None, :]
    im = im + img_mean_4_dataset[None, None, None, :]
    im = np.ascontiguousarray(im, dtype=np.uint8)
    result = np.zeros(probs.shape)
    for i in range(batchsize):
        result[i] = crf_inference(im[i], probs[i])

    result[result < 0.0001] = 0.0001
    result = result / np.sum(result, axis=1, keepdims=True)

    result = np.log(result)

    return result


def generate_supervision_by_so(feature, label, cues, mask, pred, knn_matrix):
    batchsize, class_num, h, w = pred.shape
    supervision = cues.clone()

    for i in range(batchsize):
        label_class = torch.nonzero(label[i])
        markers_new = np.zeros((h, w), dtype=np.float32)
        markers_new.fill(NUM_CLASSES)
        pos = np.where(cues[i].numpy() == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        knn_matrix_img = knn_matrix[i]
        for c in (label_class):
            c_c = c[0].numpy()
            pred_c = pred[i][c_c]

            # label_new = maxflow_dgcn_cpp.forward(markers_new, pred_c, knn_matrix_img, c_c, i,)
            # supervision[i][c_c] = torch.from_numpy(
            #     np.where(pred_c > 0.7, label_new.astype(int).reshape(41, 41),
            #              supervision[i][c_c])).float()

            # supervision[i][c[0]] = torch.from_numpy(label_new.reshape(h, w))

            supervision[i][c_c] = torch.from_numpy(maxflow_dgcn_cpp.forward(markers_new, pred_c, knn_matrix_img, c_c, i,))

    return supervision


def dgcn_generate_supervision(feature, label, cues, mask, pred, knn_matrix):
    '''

    Args:
        feature: (8, 1280, 41, 41)
        label: torch.Size([8, 21]) || :0 is background
        cues: torch.Size([8, 21, 41, 41])
        mask: none
        pred: (8, 21, 41, 41)
        knn_matrix: (8, 1681, 1681) || 41 * 41 = 1681

    Returns:
        supervision: torch.Size([8, 21, 41, 41])

    '''
    batchsize, class_num, h, w = pred.shape
    Y = torch.zeros(batchsize, class_num, h, w)
    supervision = cues.clone()

    for i in range(batchsize):
        # get the index of the non-zero class value
        label_class = torch.nonzero(label[i]).to(cues.device)
        markers_new = np.zeros((h, w))
        # class_num is 21 / 2
        markers_new.fill(class_num)
        pos = np.where(cues[i].numpy() == 1)
        # fill the correct position the class label
        markers_new[pos[1], pos[2]] = pos[0]
        markers_new_flat = markers_new.reshape(h * w)
        for c in (label_class):
            # get the exact class index
            c_c = c[0].numpy()
            # get feature of the exact one in a batch and transpose its shape
            # feature.shape[1] is the channel num
            # feature_c = feature[i].reshape(feature.shape[1], h * w).transpose(1, 0)
            # get prediction of the exact class in a batch
            pred_c = pred[i][c[0]]
            pred_c_flat = pred_c.flatten()
            # construct the maxflow Graph
            g = maxflow.Graph[float]()
            # every pixel must be a node
            nodes = g.add_nodes(h * w)
            # get the position where the cues belong to an exact class
            # ====================Foreground Class(20)====================
            pos = np.where(markers_new_flat == c_c)
            # position 0 is represent the row
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 0, 10)
                # knn matrix's shape (6, 1681, 1681)
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            # ====================Uncertain Class====================
            pos = np.where(markers_new_flat == class_num)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], -np.log10(pred_c_flat[node_i]), -np.log10(1 - pred_c_flat[node_i]))
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            # ====================Background Class(1)====================
            pos = np.where((markers_new_flat != class_num) & (markers_new_flat != c_c))
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 10, 0)
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)

            flow = g.maxflow()
            node_ids = np.arange(h * w)
            label_new = g.get_grid_segments(node_ids)

            supervision[i][c[0]] = torch.from_numpy(
                np.where(pred_c > 0.7, label_new.astype(int).reshape(h, w), supervision[i][c[0]])).float()

    return supervision

import threading
NUM_CLASSES = 21

def generate_supervision_multi_threads(feature, label, cues, mask, pred, knn_matrix):
    batchsize, class_num, h, w = pred.shape
    supervision = cues.clone()

    class GraphCutThread(threading.Thread):

        def __init__(self, threadID, markers_new, pred_c, knn_matrix_img, c_c, i):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.markers_new_flat = markers_new.flatten()
            self.pred_c_flat = pred_c.flatten()
            self.pred_c = pred_c
            self.knn_matrix_img = knn_matrix_img
            self.c_c = c_c
            self.i = i

        def run(self):
            g = maxflow.Graph[float]()
            nodes = g.add_nodes(41 * 41)
            pos = np.where(self.markers_new_flat == self.c_c)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 0, 10)
                k_neighbor = np.where(self.knn_matrix_img[node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            pos = np.where(self.markers_new_flat == NUM_CLASSES)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], -np.log10(self.pred_c_flat[node_i]),
                            -np.log10(1 - self.pred_c_flat[node_i]))
                k_neighbor = np.where(self.knn_matrix_img[node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            pos = np.where((self.markers_new_flat != NUM_CLASSES) & (self.markers_new_flat != self.c_c))
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 10, 0)
                k_neighbor = np.where(self.knn_matrix_img[node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)

            node_ids = np.arange(41 * 41)
            label_new = g.get_grid_segments(node_ids)

            print("开启线程： " + self.name)
            # 获取锁，用于线程同步
            # threadLock.acquire()
            supervision[self.i][self.c_c] = torch.from_numpy(
                np.where(self.pred_c > 0.7, label_new.astype(int).reshape(41, 41),
                         supervision[self.i][self.c_c])).float()

            # 释放锁，开启下一个线程
            # threadLock.release()

    def call_cpp_extension(lon1, lat1, lon2, lat2, test_cnt):
        # res = None
        threadLock.acquire()
        supervision[test_cnt][lat2] = torch.from_numpy(maxflow_dgcn_cpp.forward(lon1, lat1, lon2, lat2, test_cnt))
        threadLock.release()
        # print(res)

    threadLock = threading.Lock()
    threads = []

    threadID = 0
    for i in range(batchsize):
        label_class = torch.nonzero(label[i])
        markers_new = np.zeros((h, w), dtype=np.float32)
        markers_new.fill(NUM_CLASSES)
        pos = np.where(cues[i].numpy() == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        knn_matrix_img = knn_matrix[i]
        for c in (label_class):
            c_c = c[0].numpy()
            pred_c = pred[i][c_c]
            # multi threads
            # t = GraphCutThread(threadID, markers_new, pred_c, knn_matrix_img, c_c, i)
            t = threading.Thread(target=call_cpp_extension,
                args=(markers_new, pred_c, knn_matrix_img, c_c, i,))
            # t.start()
            threads.append(t)
            threadID = threadID + 1

    for t in threads:
        t.setDaemon(True)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()
    # print("退出主线程")

    return supervision


def dgcn_generate_supervision_torch(feature, label, cues, mask, pred, knn_matrix):
    '''

    Args:
        feature: (8, 1280, 41, 41)
        label: torch.Size([8, 21]) || :0 is background
        cues: torch.Size([8, 21, 41, 41])
        mask: none
        pred: (8, 21, 41, 41)
        knn_matrix: (8, 1681, 1681) || 41 * 41 = 1681

    Returns:
        supervision: torch.Size([8, 21, 41, 41])

    '''
    batchsize, class_num, h, w = pred.shape
    Y = torch.zeros(batchsize, class_num, h, w)
    supervision = cues.clone()

    for i in range(batchsize):
        # get the index of the non-zero class value
        label_class = torch.nonzero(label[i])
        markers_new = torch.zeros((h, w))
        # class_num is 21 / 2
        markers_new.fill(class_num)
        pos = torch.where(cues[i] == 1)
        # fill the correct position the class label
        markers_new[pos[1], pos[2]] = pos[0]
        markers_new_flat = markers_new.reshape(h * w)
        for c in (label_class):
            # get the exact class index
            c_c = c[0].numpy()
            # get feature of the exact one in a batch and transpose its shape
            # feature.shape[1] is the channel num
            # feature_c = feature[i].reshape(feature.shape[1], h * w).transpose(1, 0)
            # get prediction of the exact class in a batch
            pred_c = pred[i][c[0]]
            pred_c_flat = pred_c.flatten()
            # construct the maxflow Graph
            g = maxflow.Graph[float]()
            # every pixel must be a node
            nodes = g.add_nodes(h * w)
            # get the position where the cues belong to an exact class
            # ====================Foreground Class(20)====================
            pos = torch.where(markers_new_flat == c_c)
            # position 0 is represent the row
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 0, 10)
                # knn matrix's shape (6, 1681, 1681)
                k_neighbor = torch.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            # ====================Uncertain Class====================
            pos = torch.where(markers_new_flat == class_num)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], -torch.log10(pred_c_flat[node_i]), -torch.log10(1 - pred_c_flat[node_i]))
                k_neighbor = torch.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            # ====================Background Class(1)====================
            pos = torch.where((markers_new_flat != class_num) & (markers_new_flat != c_c))
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 10, 0)
                k_neighbor = torch.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)

            flow = g.maxflow()
            node_ids = torch.arange(h * w)
            label_new = g.get_grid_segments(node_ids)

            supervision[i][c[0]] = torch.from_numpy(
                torch.where(pred_c > 0.7, label_new.astype(int).reshape(h, w), supervision[i][c[0]])).float()

    return supervision


def dgcn_softmax(preds, min_prob):
    preds_max = torch.max(preds, dim=1, keepdim=True)
    preds_exp = torch.exp(preds - preds_max[0])
    probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True)
    min_prob = torch.ones((probs.shape), device=min_prob.device) * min_prob
    probs = probs + min_prob
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs


def dgcn_constrain_loss(probs, crf):
    probs_smooth = torch.exp(torch.from_numpy(crf)).float()
    # it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    #   probs_smooth = torch.tensor(probs_smooth, device=probs.device)
    probs_smooth = probs_smooth.to(probs.device)
    loss = torch.mean(torch.sum(probs_smooth * torch.log(clip(probs_smooth / probs, 0.05, 20)), dim=1))
    return loss


def dgcn_cal_seeding_loss(pred, label):
    device = pred.device
    pred_bg = pred[:, 0, :, :]
    labels_bg = label[:, 0, :, :].float().to(device)
    pred_fg = pred[:, 1:, :, :]
    labels_fg = label[:, 1:, :, :].float().to(device)

    count_bg = torch.sum(torch.sum(labels_bg, dim=2, keepdim=True), dim=1, keepdim=True)
    count_fg = torch.sum(torch.sum(torch.sum(labels_fg, dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)

    sum_bg = torch.sum(torch.sum(labels_bg * torch.log(pred_bg), dim=2, keepdim=True), dim=1, keepdim=True)
    sum_fg = torch.sum(torch.sum(torch.sum(labels_fg * torch.log(pred_fg), dim=3, keepdim=True), dim=2, keepdim=True),
                       dim=1, keepdim=True)
    loss_1 = -(sum_bg / torch.max(count_bg, torch.tensor(0.0001, device=device))).mean()
    loss_2 = -(sum_fg / torch.max(count_fg, torch.tensor(0.0001, device=device))).mean()
    loss_balanced = loss_1 + loss_2
    return loss_balanced


def clip(x, min, max):
    x_min = x < min
    x_max = x > max
    y = torch.mul(torch.mul(x, (~x_min).float()), (~x_max).float()) + ((x_min.float()) * min) + (x_max * max).float()
    return y


def dgcn_get_cues_from_seg_gt(gt_semantic_seg, cues_shape):
    B, K, H, W = cues_shape
    cues = torch.zeros(cues_shape)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                k = gt_semantic_seg[b][0][h][w].int()
                if k == 255:
                    continue
                cues[b][k][h][w] = 1.0
    return cues


def dgcn_get_cues_from_seg_gt_tensor(gt_semantic_seg, cues_shape):
    B, K, H, W = cues_shape
    cues = torch.zeros(cues_shape, dtype=torch.float32, device=gt_semantic_seg.device)

    for c in range(K):
        pos = torch.where(gt_semantic_seg == c)
        cues[pos[0], pos[1] + c, pos[2], pos[3]] = 1

    return cues


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    # 如果你说的是recall和precision假设一共有10篇文章，里面4篇是你要找的。
    # 根据你某个算法，你认为其中有5篇是你要找的，但是实际上在这5篇里面，只有3篇是真正你要找的。
    # 那么你的这个算法的precision是3/5=60%，
    # 也就是，你找的这5篇，有3篇是真正对的这个算法的recall是3/4=75%，
    # 也就是，一共有用的这4篇里面，你找到了其中三篇。请自行归纳总结。
    #
    # true positive : 3
    # false positive : 2
    # false negative : 1

    #             prediction
    #           True     False
    #     True   TP       FN
    # GT
    #     False  FP       TN

    # precision = true positive / (true positive + false positive)

    # recall = true positive / (true positive + false negative)
    def precision(self):
        recall = 0.0
        for i in xrange(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    #
    def recall(self):
        accuracy = 0.0
        for i in xrange(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    h, w = img.shape[:2]
    class_num, h, w = probs.shape
    n_labels = class_num

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80 / scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def save_seg_map_as_each_numpy(output_seg, save_path='./tmp'):
    assert output_seg.shape.size == 4

    for each_output in output_seg:
        each_output_np = each_output.numpy()
        np.save

