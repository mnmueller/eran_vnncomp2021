# import os
# import csv
# import numpy as np
# import argparse
# import time
# from multiprocessing import Process, Pipe, cpu_count
# import tensorflow as tf
import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from utils import evaluate_cstr, translate_constraints_to_label

# if tf.__version__[0] == '2':
#     import graph_def_editor as ge
# else:
#     from tensorflow.contrib import graph_editor as ge

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--model', type=str, required=True, help='Path to model')
# parser.add_argument('--epsilon', type=float, required=True, help='Epsilon')
# parser.add_argument('--pgd_step', type=float, required=True, help='Epsilon')
# parser.add_argument('--im', type=int, required=True, help='Image number')
# parser.add_argument('--it', type=int, default=500, help='Iterations')
# parser.add_argument('--mean', type=float, default=0, help='Mean')
# parser.add_argument('--std', type=float, default=1, help='Std')
# parser.add_argument('--threads', type=int, default=None, help='Number of threads')
# args = parser.parse_args()


# def create_pgd_graph(lb, ub, sess, g, tf_input_old, tf_input_new, tf_var, tf_output, target):
#     # Replace graph
#     tf_output = ge.graph_replace(tf_output, {tf_input_old: g[tf_input_new.name]})
#     sess = tf.compat.v1.Session(graph=g.to_tf_graph())
#     with sess.as_default():
#         with sess.graph.as_default():
#             tf_image = sess.graph.get_tensor_by_name(tf_input_new.name)
#             tf_output = sess.graph.get_tensor_by_name(tf_output.name)
#             # Output diversification
#             tf_dir = tf.compat.v1.placeholder(shape=(tf_output.shape[1]), dtype=tf.compat.v1.float32)
#             tf_eps_init = tf.compat.v1.placeholder(shape=tf_image.shape, dtype=tf.compat.v1.float32)
#             tf_init_error = tf.compat.v1.reduce_sum(tf_dir * tf_output)
#             tf_init_grad = tf.compat.v1.gradients(tf_init_error, [tf_image])[0]
#             tf_train_init = tf_image + tf_eps_init * tf.compat.v1.sign(tf_init_grad)
#             tf_train_init = tf.compat.v1.assign(tf_var, tf_train_init)
#
#             # PGD
#             tf_train_error = tf.compat.v1.keras.utils.to_categorical(target, num_classes=tf_output.shape[-1])
#             tf_eps_pgd = tf.compat.v1.placeholder(shape=tf_image.shape, dtype=tf.compat.v1.float32)
#             tf_train_error = tf.compat.v1.keras.losses.categorical_crossentropy(tf_train_error, tf_output, from_logits=True)
#             tf_train_grad = tf.compat.v1.gradients(tf_train_error, [tf_image])[0]
#             tf_train_pgd = tf_image - tf_eps_pgd * tf.compat.v1.sign(tf_train_grad)
#             tf_train_pgd = tf.compat.v1.assign(tf_image, tf_train_pgd)
#
#             # Clip
#             tf_train_clip = tf.compat.v1.clip_by_value(tf_image, lb, ub)
#             tf_train_clip = tf.compat.v1.assign(tf_image, tf_train_clip)
#
#             # Seed
#             tf_seed_pl = tf.compat.v1.placeholder(shape=tf_image.shape, dtype=tf.compat.v1.float64)
#             tf_seed = tf.compat.v1.assign(tf_image, tf_seed_pl)
#
#             return sess, tf_image, tf_dir, tf_seed_pl, tf_eps_init, tf_eps_pgd, tf_output, tf_train_init, tf_train_pgd, tf_train_clip, tf_seed

# def pgd(sess, lb, ub, tf_image, tf_dir, tf_seed_pl, tf_eps_init, tf_eps_pgd,
#         tf_output, tf_train_init, tf_train_pgd, tf_train_clip, tf_seed,
#         eps_init, eps_pgd, odi_its, pgd_its):
#     seed = np.random.uniform(lb, ub, size=lb.shape)
#     d = np.random.uniform(-1, 1, size=(tf_output.shape[1]))
#
#     sess.run(tf_seed, feed_dict={tf_seed_pl: seed})
#     for i in range(odi_its):
#         sess.run(tf_train_init, feed_dict={tf_dir: d, tf_eps_init: eps_init})
#         sess.run(tf_train_clip)
#     seed = sess.run(tf_image)
#
#     sess.run(tf_seed, feed_dict={tf_seed_pl: seed})
#     for i in range(pgd_its):
#         sess.run(tf_train_pgd, feed_dict={tf_eps_pgd: eps_pgd})
#         sess.run(tf_train_clip)
#     seed = sess.run(tf_image)
#
#     return seed

# def normalize(img, mean, std):
#     img = img - mean
#     img = img / std
#     return img


# def denormalize(img, mean, std):
#     img = img * std
#     img = img + mean
#     return img



# def create_pool(corr_label, threads, model, specLB, specUB, pgd_args):
#     conns = []
#     procs = []
#     parent_pid = os.getpid()
#     proc_id = 0
#     for cpu in range(10):
#         if corr_label == cpu:
#             continue
#         parent_conn, child_conn = Pipe()
#         conns.append(parent_conn)
#         p = Process(target=thread_fun, args=(proc_id % threads, cpu, child_conn, model, specLB, specUB, pgd_args))
#         p.start()
#         procs.append(p)
#         proc_id += 1
#     return conns, procs


# def thread_fun(proc_id, i, conn, model, specLB, specUB, pgd_args):
#     print('Proc', proc_id)
#     os.sched_setaffinity(0, [proc_id])
#
#     model_path = model
#
#     sess = tf.compat.v1.Session()
#     tf.compat.v1.disable_eager_execution()
#     with tf.compat.v1.gfile.FastGFile(model_path, 'rb') as f:
#         graph_def = tf.compat.v1.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.compat.v1.import_graph_def(graph_def, name='')
#     tf_out = sess.graph.get_operations()[-1].outputs[0]
#     tf_in = sess.graph.get_tensor_by_name('input.1:0')
#     tf_in_new_1 = tf.compat.v1.placeholder(shape=tf_in.shape, dtype=tf.float64, name='x')
#     tf_in_1 = tf.compat.v1.reshape(tf_in_new_1, tf_in.shape)
#     tf_in_1 = tf.compat.v1.cast(tf_in_1, tf.float32)
#     tf_in_new_2 = tf.compat.v1.Variable(specLB.reshape(tf_in.shape), trainable=True, dtype=tf.compat.v1.float32)
#     tf_in_2 = tf.compat.v1.reshape(tf_in_new_2, tf_in.shape)
#     tf_in_2 = tf.compat.v1.cast(tf_in_2, tf.float32)
#
#     g = ge.Graph(sess.graph.as_graph_def())
#     tf_output = ge.graph_replace(g[tf_out.name], {g[tf_in.name]: g[tf_in_1.name]})
#     # tf_output = tf.compat.v1.cast(tf_output, tf.float64)
#     # sess.close()
#     pgd_obj = create_pgd_graph(specLB, specUB, sess, g, g[tf_in_1.name], tf_in_2, tf_in_new_2, tf_output, i)
#
#     sess = pgd_obj[0]
#     pgd_obj = pgd_obj[1:]
#
#     for j in range(pgd_args[-1]):
#         ex = pgd(sess, specLB, specUB, *pgd_obj, *pgd_args)
#         status = conn.poll(0.001)
#         if status:
#             if conn.recv() == 'kill':
#                 return
#         if np.argmax(sess.run(tf_output, feed_dict={tf_in_new_1: ex})) == i:
#             conn.send((i, ex))
#             while True:
#                 status = conn.poll(1)
#                 if status:
#                     if conn.recv() == 'kill':
#                         return
#     conn.send((i, False))
#
#     while True:
#         status = conn.poll(1)
#         if status:
#             if conn.recv() == 'kill':
#                 return
#
#
# # def apply_attack(model, specLB, specUB, corr_label, means, stds, threads=None, pgd_step=0.1):
# #     eps = specUB - specLB
# #
# #     pgd_args = (pgd_step * eps, pgd_step * eps, 5, 100)
# #
# #     if threads is None:
# #         threads = cpu_count()
# #
# #     start = time.time()
# #     conns, procs = create_pool(corr_label, threads, model, specLB, specUB, pgd_args)
# #
# #     mapping = []
# #     for conn in range(len(conns)):
# #         mapping.append(True)
# #
# #     while True:
# #         if not np.any(mapping):
# #             break
# #         for i in range(len(conns)):
# #             conn = conns[i]
# #             if mapping[i]:
# #                 status = conn.poll(0.1)
# #                 if status:
# #                     res = conn.recv()
# #                     mapping[i] = False
# #                     conn.send('kill')
# #                     if not (res[1] is False):
# #                         print('Attack found for', res[0], ':')
# #                         ex = denormalize(res[1], means, stds)
# #                         print(ex)
# #                         for i in range(len(conns)):
# #                             if mapping[i]:
# #                                 conn = conns[i]
# #                                 conn.send('kill')
# #                         for proc in procs:
# #                             proc.join()
# #                         end = time.time()
# #                         print(end - start, "seconds")
# #                         exit()
# #                     # else:
# #                     # print( 'No attacks for', res[0] )
# #
# #     print('No attacks found')
# #     end = time.time()
# #     print(end - start, "seconds")


def margin_loss(logits, y):
    logit_org = logits.gather(1,y.view(-1,1))
    logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    loss = -logit_org + logit_target
    loss = torch.sum(loss)
    return loss


def constraint_loss(logits, constraints, and_idx=None):
    loss = 0
    for i, or_list in enumerate(constraints):
        or_loss = 0
        for cstr in or_list:
            if cstr[0] == -1:
                or_loss += -logits[:, cstr[1]]
            elif cstr[1] == -1:
                or_loss += logits[:, cstr[0]]
            else:
                or_loss += logits[:, cstr[0]] - logits[:, cstr[1]]
        if and_idx is not None:
            loss += torch.where(and_idx == i, or_loss, torch.zeros_like(or_loss))
        else:
            loss += or_loss
    return -loss


class step_lr_scheduler:
    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma**(np.floor(self.current_step/self.interval))
        else:
            phase = len([x for x in self.interval if self.current_step>=x])
            return self.initial_step_size * self.gamma**(phase)


def torch_whitebox_attack(model, device, sample, constraints, specLB, specUB, input_nchw=True, restarts=1):
    input_shape = list(sample.shape)
    input_shape = ([1] if len(input_shape) in [3,1] else []) + input_shape
    nhwc_shape = input_shape[:-3] + input_shape[-2:] + input_shape[-3:-2]
    specLB_t = torch.tensor(specLB.reshape(input_shape if input_nchw else nhwc_shape), dtype=torch.float64)
    specUB_t = torch.tensor(specUB.reshape(input_shape if input_nchw else nhwc_shape),dtype=torch.float64)
    sample = sample.reshape(input_shape if input_nchw else nhwc_shape)
    if len(input_shape)==4:
        specLB_t = specLB_t.permute((0, 1, 2, 3) if input_nchw else (0, 3, 1, 2)).to(device)
        specUB_t = specUB_t.permute((0, 1, 2, 3) if input_nchw else (0, 3, 1, 2)).to(device)
        sample = sample.permute((0, 1, 2, 3) if input_nchw else (0, 3, 1, 2))
    X = Variable(sample, requires_grad=True).to(device)

    if np.prod(input_shape)<10:
        ODI_num_steps = 0
    else:
        ODI_num_steps = 10

    adex, worst_x = _pgd_whitebox(model, X, constraints,
                         specLB_t,
                         specUB_t,
                         device, lossFunc="GAMA", restarts=restarts, ODI_num_steps=ODI_num_steps)
    if adex is None:
        adex, _ = _pgd_whitebox(model, X, constraints,
                             specLB_t,
                             specUB_t,
                             device, lossFunc="margin", restarts=restarts, ODI_num_steps=ODI_num_steps)
    if len(input_shape)==4:
        if adex is not None:
            adex = [adex[0][0].transpose((0, 1, 2) if input_nchw else (1, 2, 0))]
        if worst_x is not None:
            worst_x = worst_x.transpose((0, 1, 2) if input_nchw else (1, 2, 0))

    if adex is not None:
        assert (adex[0].flatten()>=specLB).all() and (adex[0].flatten()<=specUB).all()
        print("Adex found via attack")
    else:
        assert (worst_x.flatten()>=specLB).all() and (worst_x.flatten()<=specUB).all()
        print("No adex found via attack")
    return adex, worst_x


def _pgd_whitebox(model, X, constraints, specLB, specUB, device, num_steps=200, step_size=0.2,
                  ODI_num_steps=10, ODI_step_size=1., batch_size=50, lossFunc="margin", restarts=1):
    out_X = model(X).detach()
    adex = None
    worst_x = None
    best_loss = torch.tensor(-np.inf)

    y = translate_constraints_to_label([constraints])[0]

    for _ in range(restarts):
        if adex is not None: break
        X_pgd = Variable(X.data.repeat((batch_size,) + (1,) * (X.dim() - 1)), requires_grad=True).to(device)
        randVector_ = torch.ones_like(model(X_pgd)).uniform_(-1, 1) #torch.FloatTensor(*model(X_pgd).shape).uniform_(-1.,1.).to(device)
        random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5)*(specUB-specLB) #torch.FloatTensor(*X_pgd.shape).uniform_(-0.5, 0.5).to(device)*(specUB-specLB)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        lr_scale = (specUB-specLB)/2
        lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(0.5*num_steps), np.ceil(0.8*num_steps), np.ceil(0.9*num_steps)])
        gama_lambda = 10

        for i in range(ODI_num_steps + num_steps+1):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                out = model(X_pgd)

                cstrs_hold = evaluate_cstr(constraints, out.detach(), torch_input=True)
                if not(cstrs_hold.all()):
                    adv_idx = (~cstrs_hold.cpu()).nonzero(as_tuple=False)[0].item()
                    adex = X_pgd[adv_idx:adv_idx+1]
                    assert not evaluate_cstr(constraints, model(adex), torch_input=True)[0], f"{model(adex)},{constraints}"
                    # print("Adex found via attack")
                    return [adex.detach().cpu().numpy()], None
                if i == ODI_num_steps + num_steps:
                    # print("No adex found via attack")
                    adex = None
                    max_loss = torch.max(loss).item()
                    if max_loss > best_loss:
                        best_loss = max_loss
                        worst_x = X_pgd[torch.argmax(loss)].detach().cpu().numpy()
                    break

                if i < ODI_num_steps:
                    loss = (out * randVector_).sum()
                elif lossFunc == 'xent':
                    loss = nn.CrossEntropyLoss()(out, torch.tensor([y]*out.shape[0], dtype=torch.long))
                elif lossFunc == "margin":
                    and_idx = np.arange(len(constraints)).repeat(np.floor(batch_size/len(constraints)))
                    and_idx = torch.tensor(np.concatenate([and_idx, np.arange(batch_size-len(and_idx))], axis=0)).to(device)
                    loss = constraint_loss(out, constraints, and_idx=and_idx).sum()
                elif lossFunc == "GAMA":
                    and_idx = np.arange(len(constraints)).repeat(np.floor(batch_size / len(constraints)))
                    and_idx = torch.tensor(np.concatenate([and_idx, np.arange(batch_size - len(and_idx))], axis=0)).to(device)
                    out = torch.softmax(out,1)
                    loss = (constraint_loss(out, constraints, and_idx=and_idx) + (gama_lambda * (out_X-out)**2).sum(dim=1)).sum()
                    gama_lambda *= 0.9

            loss.backward()
            if i < ODI_num_steps:
                eta = ODI_step_size * lr_scale * X_pgd.grad.data.sign()
            else:
                eta = lr_scheduler.get_lr() * lr_scale * X_pgd.grad.data.sign()
                lr_scheduler.step()
            X_pgd = Variable(torch.minimum(torch.maximum(X_pgd.data + eta, specLB), specUB), requires_grad=True)
    return adex, worst_x


# if __name__ == "__main__":
#
#     csvfile = open('/home/mark/Projects/ERAN/eran/attacks/mnist_test_comp.csv', 'r')
#     tests = csv.reader(csvfile, delimiter=',')
#     tests = list(tests)
#     test = tests[0]
#     epsilon = 0.3
#     means = [0.1307]
#     stds = [0.3081]
#     image= np.float64(test[1:len(test)])/255.0
#     corr_label = int(test[0])
#     specLB = np.copy(image)
#     specUB = np.copy(image)
#     specLB -= epsilon
#     specUB += epsilon
#     specLB = np.maximum( 0, specLB )
#     specUB = np.minimum( 1, specUB )
#     specLB = normalize( specLB, means, stds)
#     specUB = normalize( specUB, means, stds)
#     image = normalize( image, means, stds)
#     model = "/home/mark/Projects/ERAN/eran/attacks/mnist_0.3.pb"
#
#     apply_attack(model, specLB, specUB, corr_label, means, stds, threads=None, pgd_step=0.1)