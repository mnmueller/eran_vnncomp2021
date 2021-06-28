import os
import sys

sys.path.insert(0, '../ELINA/python_interface/')
sys.path.insert(0, '../deepg/code/')

import re
import torch
import torch.nn as nn
import pickle as pkl
import csv
import tensorflow as tf
# from onnx_translator import read_onnx_net
from read_net_file import read_tensorflow_net
from eran import ERAN
import onnx
import numpy as np


# for x in new_nets:
#     net_name = re.match(".*\/([a-z,A-z,_,0-9,\.]*)\/[a-z,A-z,_,0-9,\.]*",x).group(1) + "_" + re.match(".*\/([a-z,A-z,_,0-9,\.]*)",x).group(1)
#     os.system(f"wget -O /home/mark/data/nets/temp/{net_name} {x}")


def run(dataset, last_i=0, convert=True):
    if dataset == "mnist":
        net_list = mnist_nets
    elif dataset == "cifar10":
        net_list = cifar_nets
    else:
        assert False

    for i in range(len(net_list)):
        if i < last_i:
            continue

        x = net_list[i]
        net_name = re.match(".*\/([a-z,A-z,_,0-9,\.]*)",x).group(1)

        # os.system(f"wget -O /home/mark/data/nets/cifar10/{net_name} {x}")
        print(f"Converting network: {net_name}")

        if convert:
            netname = os.path.join(f"/home/mark/data/nets/{dataset}", net_name)
            convert_net(netname, dataset)
        else:
            net_name = re.match("([a-z,A-z,_,0-9,\.]*)\.[a-z]*", net_name).group(1) + ".onnx"
            netname = os.path.join(f"/home/mark/data/nets/{dataset}", net_name)
            buzz_net(netname, dataset)


def get_tests(dataset, geometric):
    if geometric:
        csvfile = open('../deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        csvfile = open('../data/{}_test.csv'.format(dataset), 'r')
    tests = csv.reader(csvfile, delimiter=',')
    return tests


def normalize(image, means, stds, dataset, domain, is_conv):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds != None:
                image[i] /= stds[i]
    elif dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0]) / stds[0]
    elif (dataset == 'cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0]) / stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1]) / stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2]) / stds[2]
            count = count + 1

        is_gpupoly = (domain == 'gpupoly' or domain == 'refinegpupoly')
        if is_conv and not is_gpupoly:
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count + 1
                image[i + 1024] = tmp[count]
                count = count + 1
                image[i + 2048] = tmp[count]
                count = count + 1


def read_onnx_net(net_file):
    onnx_model = onnx.load(net_file)
    onnx.checker.check_model(onnx_model)

    is_conv = False

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            is_conv = True
            break

    return onnx_model, is_conv


class Normalize(torch.nn.Module):
    def __init__(self, means, stds, channel_dim):
        super(Normalize, self).__init__()
        target_shape = 4 * [1]
        target_shape[channel_dim] = len(means)
        self.means = torch.tensor(means, dtype=dtype).reshape(target_shape)
        self.stds = torch.tensor(stds, dtype=dtype).reshape(target_shape)

    def forward(self, x):
        return (x - self.means) / self.stds


def create_torch_net_new(operations, resources, means, stds, input_shape, is_nchw, dtype=torch.float64):
    layers = []
    conv_section = True
    skip_next = False
    input_shape = [input_shape[-1], input_shape[-3], input_shape[-2]] if len(input_shape)>=3 else [input_shape[-1]]
    layers_dict = {resources[0]["deeppoly"][-2]: []}
    predecessor_dict = {resources[0]["deeppoly"][-2]: []}

    if means is not None and stds is not None:
        layers_dict[resources[0]["deeppoly"][2]] += [Normalize(means, stds, 1)]

    for i in range(len(operations)):
        op = operations[i]
        res = resources[i]["deeppoly"]
        inputs = res[-3]
        output = res[-2]

        if op == "Placeholder": continue
        if skip_next:
            skip_next = False
            continue

        if len(inputs) == 1:
            layers = layers_dict[inputs[0]].copy()
            # predecessor_dict[output] = predecessor_dict[inputs[0]] + [inputs[0]]

        res_b = None

        if op in ["Conv2D", "Conv", "MatMul"]:
            if len(operations) > i + 1 and operations[i + 1] in ["Add", "BiasAdd"]:
                skip_next = True
                res_b = resources[i + 1]["deeppoly"]
                output = resources[i+1]["deeppoly"][-2]

        if op == "Relu":
            layers += [torch.nn.ReLU()]

        elif op == "Tanh":
            layers += [torch.nn.Tanh()]

        elif op == "Sigmoid":
            layers += [torch.nn.Sigmoid()]

        elif op in ["Conv2D", "Conv"]:
            bias = False
            if len(res) == 10:
                filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, _, _, _ = res
                if res_b is not None:
                    bias_d = torch.tensor(res_b[0], dtype=dtype)
                    bias = True
            else:
                filters, bias_d, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, _, _, _ = res
                #             filters = torch.tensor(filters,dtype=dtype)
                bias_d = torch.tensor(bias_d, dtype=dtype)
                bias = True
            filters = torch.tensor(filters, dtype=dtype)
            # if not is_nchw:
            filters = filters.permute(3, 2, 0, 1)
            if not ((pad_top == pad_bottom) and (pad_left == pad_right)):
                layers += [torch.nn.ZeroPad2d((pad_top, pad_bottom, pad_left, pad_right))]
                pad_top = 0
                pad_left = 0
            layers += [torch.nn.Conv2d(input_shape[0], filters.shape[0], filters.shape[2:], tuple(strides), padding=(pad_top, pad_left), bias=bias)]
            layers[-1].weight.data = filters
            if bias:
                layers[-1].bias.data = bias_d
            input_shape = res[-1][1:] if is_nchw else res[-1][-1:] + res[-1][1:-1]

        elif op in ["MatMul", "Gemm"]:
            bias = False
            weights = torch.tensor(res[0], dtype=dtype)
            if res_b is not None:
                bias_d = torch.tensor(res_b[0], dtype=dtype)
                bias = True
            elif op == "Gemm":
                bias_d = torch.tensor(res[1], dtype=dtype)
                bias = True
            if conv_section:
                conv_section = False
                layers += [torch.nn.Flatten()]
                if not is_nchw and len(input_shape)>1:
                    # print(input_shape)
                    idx = torch.arange(weights.shape[1]).view(input_shape[1], input_shape[2], input_shape[0]).permute(2, 0, 1).flatten()
                    # idx = torch.arange(weights.shape[1]).view(h_current, w_current, c_current).permute(2, 0, 1).flatten()
                    assert len(idx) == weights.shape[1]
                    weights = weights.permute(1, 0)[idx].permute(1, 0)
            layers += [torch.nn.Linear(weights.shape[1], weights.shape[0], bias=bias)]
            layers[-1].weight.data = weights
            if bias:
                layers[-1].bias.data = bias_d
            input_shape = res[-1][1:]
        elif op in ["MaxPool", "AvgPool", "AveragePool"]:
            image_shape, kernel_shape, strides, pad_top, pad_left, pad_bottom, pad_right, _, _, _ = res
            if not ((pad_top == pad_bottom) and (pad_left == pad_right)):
                layers += [torch.nn.ZeroPad2d((pad_top, pad_bottom, pad_left, pad_right))]
                pad_top = 0
                pad_left = 0
            if op == "MaxPool":
                layers += [torch.nn.MaxPool2d(tuple(kernel_shape), tuple(strides), padding=(pad_top, pad_left))]
            else:
                layers += [torch.nn.AvgPool2d(tuple(kernel_shape), tuple(strides), padding=(pad_top, pad_left))]
            input_shape = res[-1][1:] if is_nchw else res[-1][-1:] + res[-1][1:-1]
        elif op == "Placeholder":
            pass
        elif op == "Resadd":
            layers_a = layers_dict[inputs[0]]
            layers_b = layers_dict[inputs[1]]

            last_common_pred = [x for x in predecessor_dict[inputs[0]] if x in predecessor_dict[inputs[1]]][-1]
            unique_layers_a = layers_a[len(layers_dict[last_common_pred]):]
            unique_layers_b = layers_b[len(layers_dict[last_common_pred]):]
            assert unique_layers_a == [x for x in layers_a if x not in layers_dict[last_common_pred]]
            # predecessor_dict[output] = predecessor_dict[last_common_pred] + [last_common_pred]
            layers_in = layers_dict[last_common_pred]
            layers = [Resadd(layers_in, unique_layers_a, unique_layers_b)]
            inputs = [last_common_pred]
        else:
            assert False, f"layer {op} not known"
        layers_dict[output] = layers
        predecessor_dict[output] = predecessor_dict[inputs[0]] + [inputs[0]]
    net = torch.nn.Sequential(*layers)
    return net, layers


class Resadd(nn.Module):
    def __init__(self, in_layers, layers_a, layers_b):
        super(Resadd, self).__init__()
        self.in_layers = nn.Sequential(*in_layers) if isinstance(in_layers, list) else in_layers
        self.layers_a = nn.Sequential(*layers_a) if isinstance(layers_a, list) else layers_a
        self.layers_b = nn.Sequential(*layers_b) if isinstance(layers_b, list) else layers_b

    def forward(self, x):
        x_in = self.in_layers(x)
        x_a = self.layers_a(x_in.clone())
        x_b = self.layers_b(x_in.clone())
        return x_a + x_b


def create_torch_net(operations, resources, means, stds, input_shape, is_nchw, dtype=torch.float64):
    layers = []
    conv_section = True
    skip_next = False
    input_shape = [input_shape[-1], input_shape[-3], input_shape[-2]]

    if means is not None and stds is not None:
        layers += [Normalize(means, stds, 1)]

    for i in range(len(operations)):
        if skip_next:
            skip_next = False
            continue

        op = operations[i]
        res = resources[i]["deeppoly"]
        res_b = None

        if op in ["Conv2D", "Conv", "MatMul"]:
            if len(operations) > i + 1 and operations[i + 1] == "BiasAdd":
                skip_next = True
                res_b = resources[i + 1]["deeppoly"]

        if op == "Relu":
            layers += [torch.nn.ReLU()]

        elif op == "Tanh":
            layers += [torch.nn.Tanh()]

        elif op == "Sigmoid":
            layers += [torch.nn.Sigmoid()]

        elif op in ["Conv2D", "Conv"]:
            bias = False
            if len(res) == 10:
                filters, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, _, _, _ = res
                if res_b is not None:
                    bias_d = torch.tensor(res_b[0], dtype=dtype)
                    bias = True
            else:
                filters, bias_d, image_shape, strides, pad_top, pad_left, pad_bottom, pad_right, _, _, _ = res
                #             filters = torch.tensor(filters,dtype=dtype)
                bias_d = torch.tensor(bias_d, dtype=dtype)
                bias = True
            filters = torch.tensor(filters, dtype=dtype)
            # if not is_nchw:
            filters = filters.permute(3, 2, 0, 1)
            if not ((pad_top == pad_bottom) and (pad_left == pad_right)):
                layers += [torch.nn.ZeroPad2d((pad_top, pad_bottom, pad_left, pad_right))]
                pad_top = 0
                pad_left = 0
            layers += [torch.nn.Conv2d(input_shape[0], filters.shape[0], filters.shape[2:], strides, padding=(pad_top, pad_left), bias=bias)]
            layers[-1].weight.data = filters
            if bias:
                layers[-1].bias.data = bias_d
            input_shape = res[-1][1:] if is_nchw else res[-1][-1:] + res[-1][1:-1]

        elif op in ["MatMul", "Gemm"]:
            bias = False
            weights = torch.tensor(res[0], dtype=dtype)
            if res_b is not None:
                bias_d = torch.tensor(res_b[0], dtype=dtype)
                bias = True
            elif op == "Gemm":
                bias_d = torch.tensor(res[1], dtype=dtype)
                bias = True
            if conv_section:
                conv_section = False
                layers += [torch.nn.Flatten()]
                if not is_nchw:
                    # print(input_shape)
                    idx = torch.arange(weights.shape[1]).view(input_shape[1], input_shape[2], input_shape[0]).permute(2, 0, 1).flatten()
                    # idx = torch.arange(weights.shape[1]).view(h_current, w_current, c_current).permute(2, 0, 1).flatten()
                    assert len(idx) == weights.shape[1]
                    weights = weights.permute(1, 0)[idx].permute(1, 0)
            layers += [torch.nn.Linear(weights.shape[1], weights.shape[0], bias=bias)]
            layers[-1].weight.data = weights
            if bias:
                layers[-1].bias.data = bias_d
            input_shape = res[-1][1:]
        elif op in ["MaxPool", "AvgPool"]:
            image_shape, kernel_shape, strides, pad_top, pad_left, pad_bottom, pad_right, _, _, _ = res
            if not ((pad_top == pad_bottom) and (pad_left == pad_right)):
                layers += [torch.nn.ZeroPad2d((pad_top, pad_bottom, pad_left, pad_right))]
                pad_top = 0
                pad_left = 0
            if op == "MaxPool":
                layers += [torch.nn.MaxPool2d(kernel_shape, strides, padding=(pad_top, pad_left))]
            else:
                layers += [torch.nn.AvgPool2d(kernel_shape, strides, padding=(pad_top, pad_left))]
            input_shape = res[-1][1:] if is_nchw else res[-1][-1:] + res[-1][1:-1]
        elif op == "Placeholder":
            pass
        else:
            assert False, f"layer {op} not known"
    net = torch.nn.Sequential(*layers)
    return net, layers


def get_eran_model(netname, input_size=None, write_pkl=False):

    filename, file_extension = os.path.splitext(netname)

    is_trained_with_pytorch = file_extension==".pyt"
    is_saved_tf_model = file_extension==".meta"
    is_pb_file = file_extension==".pb"
    is_tensorflow = file_extension== ".tf"
    is_onnx = file_extension == ".onnx"
    assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

    non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

    if write_pkl:
        pkl_file = re.match("(.*)\..*", netname).group(1) + ".pkl"
    else:
        pkl_file = None

    means, stds = None, None
    if is_saved_tf_model or is_pb_file:
        netfolder = os.path.dirname(netname)

        tf.logging.set_verbosity(tf.logging.ERROR)

        sess = tf.Session()
        if is_saved_tf_model:
            saver = tf.train.import_meta_graph(netname)
            saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
        else:
            with tf.gfile.GFile(netname, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.graph_util.import_graph_def(graph_def, name='')
        ops = sess.graph.get_operations()
        last_layer_index = -1
        while ops[last_layer_index].type in non_layer_operation_types:
            last_layer_index -= 1
        model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')

        eran = ERAN(model, sess, pkl_file=pkl_file)
    else:
        if is_onnx:
            model, is_conv = read_onnx_net(netname)
        else:
            model, is_conv, means, stds = read_tensorflow_net(netname, input_size, is_trained_with_pytorch, False)
        eran = ERAN(model, is_onnx=is_onnx, pkl_file=pkl_file)
    return eran, is_trained_with_pytorch, is_conv, means, stds


def buzz_net(netname, dataset):
    domain = "deeppoly"

    tests = get_tests(dataset, False)
    input_size = np.prod(tests[0].shape)

    eran, in_net_normalization, is_conv, means, stds = get_eran_model(netname, input_size, False)

    if not in_net_normalization:
        if dataset == 'mnist':
            means = [0]
            stds = [1]
        elif dataset == 'acasxu':
            means = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
            stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
        elif dataset == "cifar10":
            means = [0.4914, 0.4822, 0.4465]
            stds = [0.2023, 0.1994, 0.2010]
        else:
            means = [0]
            stds = [1]

    correct = 0
    for i, test in enumerate(tests):
        image = np.float64(test[1:len(test)]) / np.float64(255)
        y = int(test[0])
        specLB = np.copy(image)
        specUB = np.copy(image)
        normalize(specLB, means, stds, dataset, domain, is_conv)
        normalize(specUB, means, stds, dataset, domain, is_conv)
        label, nn, nlb, nub, _, _, _ = eran.analyze_box(specLB, specUB, "deeppoly", 10, 10, True)
        correct += label == y

        if i>=9:
            break
    print(f"accuracy: {correct}/{i+1}")


def convert_net(netname, dataset):
    domain = "deeppoly"
    if dataset == "cifar10":
        input_shape = (32, 32, 3)
    elif dataset == "mnist":
        input_shape = (28, 28, 1)

    tests = get_tests(dataset, False)
    input_size = np.prod(tests[0].shape)

    eran, in_net_normalization, is_conv, means, stds = get_eran_model(netname, input_size, True)

    if not in_net_normalization:
        if dataset == 'mnist':
            means = [0]
            stds = [1]
        elif dataset == 'acasxu':
            means = [1.9791091e+04, 0.0, 0.0, 650.0, 600.0]
            stds = [60261.0, 6.28318530718, 6.28318530718, 1100.0, 1200.0]
        elif dataset == "cifar10":
            means = [0.4914, 0.4822, 0.4465]
            stds = [0.2023, 0.1994, 0.2010]
        else:
            means = [0.,]
            stds = [1,]


    with open(os.path.join(re.match("(.*)\..*",netname).group(1)+".pkl"), "rb") as f:
        resources, operations = pkl.load(f)

    if "Resadd" in operations:
        print("ResNets not supported")
        return

    try:
        net, layers = create_torch_net(operations, resources, means, stds, input_shape, False)
        for i, test in enumerate(tests):
            if i>2:
                break

            image = np.float64(test[1:len(test)])/np.float64(255)
            specLB = np.copy(image)
            specUB = np.copy(image)
            normalize(specLB, means, stds, dataset, domain, is_conv)
            normalize(specUB, means, stds, dataset, domain, is_conv)

            label, nn, nlb, nub, _, _, _ = eran.analyze_box(specLB, specUB, "deeppoly", 10, 10, True)
            net_out = nlb[-1]
            torch_image = torch.tensor(image.reshape(input_shape), dtype=dtype).unsqueeze(0).permute(0,3,1,2)
            torch_out = net(torch_image).detach().cpu().numpy()

            assert np.isclose(net_out, torch_out, rtol=1e-3, atol=5e-4).all()
    except:
        try:
            net, layers = create_torch_net(operations, resources, means, stds, input_shape, True)
            for i, test in enumerate(tests):
                if i > 2:
                    break

                image = np.float64(test[1:len(test)]) / np.float64(255)
                specLB = np.copy(image)
                specUB = np.copy(image)
                normalize(specLB, means, stds, dataset, domain, is_conv)
                normalize(specUB, means, stds, dataset, domain, is_conv)

                label, nn, nlb, nub, _, _, _ = eran.analyze_box(specLB, specUB, "deeppoly", 10, 10, True)
                net_out = nlb[-1]
                torch_image = torch.tensor(image.reshape(input_shape), dtype=dtype).unsqueeze(0).permute(0, 3, 1, 2)
                torch_out = net(torch_image).detach().cpu().numpy()
                assert np.isclose(net_out, torch_out, rtol=1e-3, atol=5e-4).all()
        except:
            network_fails = True
            try:
                create_torch_net(operations, resources, means, stds, input_shape, False)
                network_fails = False
            except:
                pass
            if network_fails:
                assert False, "Network translation failed"
            else:
                assert False, "Both nchw and nhwc lead to mismatched outputs"


    model_name = os.path.join(re.match("(.*)\..*", netname).group(1))
    torch.onnx.export(net, torch_image, model_name + ".onnx", verbose=True, input_names=["input"], output_names=["output"])
    torch.save(net.state_dict(), model_name + ".pyt")
    torch.save(net, model_name + ".pynet")


if __name__ == "__main__":
    new_nets = ["https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_3_50.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_3_100.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_5_100.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_6_100.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_9_100.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_6_200.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_9_200.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnRELU__Point_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnRELU__PGDK_w_0.1_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnRELU__PGDK_w_0.3_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnSIGMOID__Point_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnSIGMOID__PGDK_w_0.1_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnSIGMOID__PGDK_w_0.3_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnTANH__Point_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnTANH__PGDK_w_0.1_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/ffnnTANH__PGDK_w_0.3_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_relu_4_1024.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__PGDK.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__PGDK_w_0.1_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__PGDK_w_0.3_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGSIGMOID__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGSIGMOID__PGDK_w_0.1_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGSIGMOID__PGDK_w_0.3_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGTANH__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGTANH__PGDK_w_0.1_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGTANH__PGDK_w_0.3_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/mnist_conv_maxpool.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convBigRELU__DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSuperRELU__DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_4_100.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_6_100.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_9_200.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnRELU__Point_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnRELU__PGDK_w_0.0078_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnRELU__PGDK_w_0.0313_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnSIGMOID__Point_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnSIGMOID__PGDK_w_0.0078_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnSIGMOID__PGDK_w_0.0313_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnTANH__Point_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnTANH__PGDK_w_0.0078_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ffnnTANH__PGDK_w_0.0313_6_500.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_relu_7_1024.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__PGDK.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convSmallRELU__DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__PGDK_w_0.0078.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__PGDK_w_0.0313.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGSIGMOID__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGSIGMOID__PGDK_w_0.0078.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGSIGMOID__PGDK_w_0.0313.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGTANH__Point.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGTANH__PGDK_w_0.0078.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGTANH__PGDK_w_0.0313.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/cifar_conv_maxpool.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convBigRELU__DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNetTiny_PGD.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNetTiny_DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet18_PGD.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet18_DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/SkipNet18_DiffAI.onnx",
                "https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/ResNet34_DiffAI.onnx"]

    # |         | Skip      | Residual | 71,650 | 9 | ReLU | DiffAI | [:arrow_down:](https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/skip__DiffAI.onnx) |

    mnist_nets = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_50.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_3_100.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_5_100.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_100.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_100.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_6_200.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_9_200.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__Point_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.1_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnRELU__PGDK_w_0.3_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__Point_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.1_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnSIGMOID__PGDK_w_0.3_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__Point_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.1_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/ffnnTANH__PGDK_w_0.3_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_relu_4_1024.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__PGDK.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSmallRELU__DiffAI.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.1.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGRELU__PGDK_w_0.3.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.1.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGSIGMOID__PGDK_w_0.3.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.1.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convMedGTANH__PGDK_w_0.3.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/mnist/mnist_conv_maxpool.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convBigRELU__DiffAI.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/convSuperRELU__DiffAI.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/mnist/skip__DiffAI.pyt", ]

    cifar_nets = ["https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_4_100.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_6_100.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_9_200.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__Point_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0078_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnRELU__PGDK_w_0.0313_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__Point_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0078_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnSIGMOID__PGDK_w_0.0313_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__Point_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0078_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ffnnTANH__PGDK_w_0.0313_6_500.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_relu_7_1024.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__PGDK.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convSmallRELU__DiffAI.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0078.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGRELU__PGDK_w_0.0313.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0078.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGSIGMOID__PGDK_w_0.0313.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__Point.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0078.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convMedGTANH__PGDK_w_0.0313.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/tensorflow/cifar/cifar_conv_maxpool.tf",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/convBigRELU__DiffAI.pyt",
                  "https://files.sri.inf.ethz.ch/eran/nets/pytorch/cifar/ResNet18_DiffAI.pyt", ]

    dtype = torch.float32

    last_i = 32
    dataset = "mnist"
    run(dataset, last_i)