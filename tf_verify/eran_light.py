"""
  Copyright 2020 ETH Zurich, Secure, Reliable, and Intelligent Systems Lab

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import sys
import os
import gzip
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
cpu_affinity = os.sched_getaffinity(0)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ELINA/python_interface/'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../deepg/code/'))
import torch
import numpy as np
from eran import ERAN
from read_net_file import read_onnx_net
from attacks import torch_whitebox_attack
import pickle as pkl
import google
import csv
from multiprocessing import Pool, Value
import time
from tqdm import tqdm
from ai_milp import verify_network_with_milp
import argparse
from config import config
import PIL
from constraint_utils import get_constraints_for_dominant_label
import re
import itertools
#from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging
import spatial
from copy import deepcopy
from onnx_translator import ONNXTranslator
from optimizer import Optimizer
from analyzer import layers
from pprint import pprint
from refine_gpupoly import refine_gpupoly_results
from utils import parse_vnn_lib_prop, translate_box_to_sample, evaluate_cstr, check_timeout
from convert_nets import read_onnx_net, create_torch_net, create_torch_net_new
import onnx

EPS = 10**(-9)
DTYPE = np.float64


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname


def generate_constraints(class_num, y):
    return [[(y,i,0)] for i in range(class_num) if i!=y]


def normalize(image, means, stds, is_nchw, input_shape):
    # normalization taken out of the network
    current_shape = image.shape
    if means is None:
        means = np.array(0)
    if stds is None:
        stds = np.array(1)
    means = means.reshape((1, -1, 1, 1)) if is_nchw else means.reshape((1, 1, 1, -1))
    stds = stds.reshape((1, -1, 1, 1)) if is_nchw else stds.reshape((1, 1, 1, -1))
    return ((image.reshape(input_shape)-means)/stds).reshape(current_shape)


def denormalize(image, means, stds, is_nchw, input_shape):
    # denormalization taken out of the network
    current_shape = image.shape
    means = 0 if means is None else (means.reshape((1,-1,1,1)) if is_nchw else means.reshape((1,1,1,-1)))
    stds = 1 if stds is None else (stds.reshape((1, -1, 1, 1)) if is_nchw else stds.reshape((1, 1, 1, -1)))
    return (image.reshape(input_shape)*stds + means).reshape(current_shape)


def vnn_lib_data_loader(vnn_lib_spec, dtype=np.float32, index_is_nchw=True, output_is_nchw=False):
    if not (os.path.isfile(vnn_lib_spec)) or (not vnn_lib_spec.endswith(".vnnlib")):
        assert False, f"Provided specification is no .vnnlib file: {vnn_lib_spec}"

    boxes, constraints = parse_vnn_lib_prop(vnn_lib_spec, dtype)
    assert all([np.all(box[0]<=box[1]) for box in boxes]), "Invalid input spec found"

    mean = None
    std = None
    if len(boxes[0][0]) == 28*28:
        input_shape = [1, 28, 28]
    elif len(boxes[0][0]) == 3*32*32:
        input_shape = [3, 32, 32]
    else:
        input_shape = [len(boxes[0][0])]
    if len(input_shape)==3 and not (index_is_nchw == output_is_nchw):
        if index_is_nchw:
            boxes = [(x.reshape(input_shape).transpose(1, 2, 0).flatten(), y.reshape(input_shape).transpose(1, 2, 0).flatten()) for x, y in boxes]
        elif output_is_nchw:
            boxes = [(x.reshape(input_shape[1:]+input_shape[:1]).transpose(2, 0, 1).flatten(),y.reshape(input_shape[1:]+input_shape[:1]).transpose(2, 0, 1).flatten()) for x, y in boxes]
    return boxes, constraints, mean, std, output_is_nchw, input_shape


def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d


def get_args(verbosity=2):
    parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--netname', type=str, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
    parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly')
    parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
    parser.add_argument('--vnnlib_spec', type=str, default=config.vnnlib_spec, help="The specification to certify in VNNLIB format")
    parser.add_argument('--instance_file', type=str, default=config.instance_file, help="A file listing specifications to certify in VNNLIB format")
    parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
    parser.add_argument('--n_test', type=int, default=config.n_test, help='Number of images to test')
    parser.add_argument('--index_is_nchw', type=str2bool, default=config.index_is_nchw,  help='Whether input indices are in nchw format')


    parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
    parser.add_argument('--timeout_final_lp', type=float, default=config.timeout_final_lp,  help='timeout for the final LP solver')
    parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
    parser.add_argument('--timeout_final_milp', type=float, default=config.timeout_final_lp,  help='timeout for the final MILP solver')
    parser.add_argument('--timeout_complete', type=float, default=config.timeout_complete,  help='Cumulative timeout for full verification')

    parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')
    parser.add_argument('--prepare', action='store_true', default=config.prepare, help='Whether to only prepare sample')
    parser.add_argument('--res_file', type=str, default=config.res_file, help='Write result to file')



    parser.add_argument('--initial_splits', type=int, default=config.initial_splits, help='Number of initial splits for low dim recursive')
    parser.add_argument('--max_split_depth', type=int, default=config.max_split_depth, help='max_split_depth for low dim recursive')
    parser.add_argument('--attack_restarts', type=int, default=config.attack_restarts, help='Number of attack batches per loss func')



    # PRIMA parameters
    parser.add_argument('--k', type=int, default=config.k, help='refine group size')
    parser.add_argument('--s', type=int, default=config.s, help='refine group sparsity parameter')
    parser.add_argument("--approx_k", type=str2bool, default=config.approx_k, help="Use approximate fast k neuron constraints")
    parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
    parser.add_argument("--max_gpu_batch", type=int, default=config.max_gpu_batch, help="maximum number of queries on GPU")

    # Refinement parameters
    parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
    parser.add_argument('--n_milp_refine', type=int, default=config.n_milp_refine, help='Number of milp refined layers')
    parser.add_argument('--max_milp_neurons', type=int, default=config.max_milp_neurons,  help='Maximum number of neurons to use for partial MILP encoding.')
    parser.add_argument('--partial_milp', type=int, default=config.partial_milp,  help='Number of layers to encode using MILP')

    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(config, k, v)
    config.json = vars(args)
    if verbosity>=1:
        pprint(config.json)

    assert (config.netname is not None) or (config.instance_file is not None), 'a network has to be provided for analysis.'
    assert (config.vnnlib_spec is not None) or (config.instance_file is not None), 'a spec has to be provided for analysis.'

    assert config.domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly', 'gpupoly', 'refinegpupoly'], \
        "domain name can be either deepzono, refinezono, deeppoly, refinepoly, gpupoly, refinegpupoly"

    return config


def init(args):
    global attack_success
    attack_success = args or attack_success


def evaluate_net(x, domain, network=None, eran=None):
    if "gpu" in domain:
        net_out = network.eval(np.array(x))[:, 0]
        label = np.argmax(net_out)
    else:
        label, _, net_out, _, _, _, _ = eran.analyze_box(x, x, 'deepzono', config.timeout_lp,
                                                          config.timeout_milp,
                                                          config.use_default_heuristic, label=0 if config.regression else -1)
        net_out = np.array(net_out[-1])
    return label, net_out


def extract_label_from_cstr(constraints, n_classes=None):
    if n_classes is None:
        n_classes = len(constraints)+1
    adv_labels = n_classes * [0]
    target = constraints[0][0][0]
    adv_labels[target] = 1
    failed = False
    for cstr in constraints:
        if not len(cstr) == 1:
            failed = True
        elif not cstr[0][0] == target:
            failed = True
        elif not cstr[0][2] == 0:
            failed = True
        if failed:
            break
        adv_labels[cstr[0][1]] = 1
    failed = failed or not all(adv_labels)
    return -1 if failed else target


def cstr_matrix_from_cstr(constraints, n_class = None, dtype = np.float32, permit_disjunctive=False):
    if (not all([len(cstr)==1 for cstr in constraints])) and not permit_disjunctive:
        return None
    if n_class is None:
        n_class = max([max(cstr[0][0], cstr[0][1]) for cstr in constraints]) + 1
    n_cstr = len([x for y in constraints for x in y])
    cstr_matrix = np.zeros((n_cstr,n_class+1), dtype=dtype)
    cstr_indices = []
    i=0
    for j, cstrs in enumerate(constraints):
        cstr_indices.append([])
        for cstr in cstrs:
            cstr_indices[j].append(i)
            if cstr[0] == -1:
                cstr_matrix[i,-1] = cstr[2]
                cstr_matrix[i, cstr[1]] = -1
            elif cstr[1] == -1:
                cstr_matrix[i, -1] = -cstr[2]
                cstr_matrix[i, cstr[0]] = 1
            else:
                cstr_matrix[i, -1] = -cstr[2]
                cstr_matrix[i, cstr[1]] = -1
                cstr_matrix[i, cstr[0]] = 1
            i += 1
    return cstr_matrix, cstr_indices


def estimate_grads(specLB, specUB, eval_instance, dim_samples=3, dtype=np.float32):
    # Estimate gradients using central difference quotient and average over dim_samples+1 in the range of the input bounds
    # Very computationally costly
    specLB = np.array(specLB, dtype=dtype)
    specUB = np.array(specUB, dtype=dtype)
    inputs = [(((dim_samples - i) * specLB + i * specUB) / dim_samples) for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB), dtype=dtype)

    for sample in range(dim_samples + 1):
        pred = eval_instance(inputs[sample])[1]
        for index in range(len(specLB)):
            if sample < dim_samples:
                l_input = [m if i != index else u for i, m, u in zip(range(len(specLB)), inputs[sample], inputs[sample+1])]
                l_input = np.array(l_input, dtype=dtype)
                l_i_pred = eval_instance(l_input)[1]
            else:
                l_i_pred = pred
            if sample > 0:
                u_input = [m if i != index else l for i, m, l in zip(range(len(specLB)), inputs[sample], inputs[sample-1])]
                u_input = np.array(u_input, dtype=dtype)
                u_i_pred = eval_instance(u_input)[1]
            else:
                u_i_pred = pred
            diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
            diffs[index] += diff
    return diffs / dim_samples


def low_dim_recursive(config, nn, specLB, specUB, domain, constraints, max_depth=10, depth=0, eran=None, network=None, relu_layers=None, verbosity=2, input_nchw=False):
    global failed_already, start_time
    def eval_instance(x):
        return evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)

    if bool(failed_already.value):
        return False, None

    if check_timeout(config, start_time, min_remaining=0.01):
        return False, None

    is_verified, adex, failed_constraints, nlb, nub, nn = verify_network(config, nn, specLB, specUB, constraints,
                                                                         domain, network, eran, relu_layers,
                                                                         find_adex=True, verbosity=verbosity,
                                                                         start_time=start_time, input_nchw=input_nchw)
    if is_verified:
        return is_verified, None
    elif depth >= max_depth:
        if (not bool(failed_already.value)) and config.complete:
            try:
                if nlb is None and network is not None:
                    layerno = 0
                    nlb = []
                    nub = []
                    gpu_spec = network.evalAffineExpr(layer=0)
                    if not (np.isclose(gpu_spec[:,0],specLB).all() and np.isclose(gpu_spec[:,1],specUB).all()):
                        network.evalAffineExpr_withProp(specLB, specUB)
                    for l in range(nn.numlayer):
                        if nn.layertypes[l] in ["FC", "Conv", ]:
                            layerno += 2
                        else:
                            layerno += 1
                        if layerno in relu_layers:
                            pre_lbi = nlb[len(nlb) - 1]
                            pre_ubi = nub[len(nub) - 1]
                            lbi = np.maximum(0, pre_lbi)
                            ubi = np.maximum(0, pre_ubi)
                        else:
                            bounds = network.evalAffineExpr(layer=layerno)
                            # bounds_2 = network.evalAffineExpr_withProp(specLB, specUB, layer=layerno)
                            # assert np.isclose(bounds_2[:,0],bounds[:,0]).all()
                            # assert np.isclose(bounds_2[:,1],bounds[:,1]).all()
                            lbi = bounds[:, 0]
                            ubi = bounds[:, 1]
                        nlb.append(lbi)
                        nub.append(ubi)
                is_verified, adex, adv_val = verify_network_with_milp(config, nn, specLB, specUB, nlb, nub, constraints,
                                                                      gpu_model="gpu" in domain, start_time=start_time,
                                                                      find_adex=True, verbosity=0, is_nchw=input_nchw)
            except Exception as ex:
                print(f"{ex}Exception occured for the following inputs:")
                print(specLB, specUB, max_depth, depth)
                raise ex
            adex_found = False
            if not is_verified:
                if adex is not None and eval_instance is not None:
                    for x in adex:
                        cex_label, cex_out = eval_instance(x)
                        adex_found = not evaluate_cstr(constraints, cex_out)
                        if adex_found:
                            if verbosity>=2:
                                print("property violated at ", x, "output_score", cex_out)
                            failed_already.value = True
                            break
            if is_verified or adex_found or time.time() - start_time - config.timeout_complete > -1:
                return is_verified, None if not adex_found else x
            else:
                max_depth += 3

    if len(specLB)>1:
        grads = estimate_grads(specLB, specUB, eval_instance).astype(specLB.dtype)
        smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])
        index = np.argmax(smears)
    else:
        index = 0

    m = (specLB[index]+specUB[index])/2
    specLB_b = specLB.copy()
    specLB_b[index] = m
    specUB_a = specUB.copy()
    specUB_a[index] = m

    result_a, adex_a = low_dim_recursive(config, nn, specLB, specUB_a,
                                         domain, constraints, max_depth=max_depth, depth=depth+1, eran=eran,
                                         network=network, relu_layers=relu_layers, verbosity=verbosity, input_nchw=input_nchw)
    if adex_a is None:
        result_b, adex_b = low_dim_recursive(config, nn, specLB_b,
                                             specUB, domain, constraints, max_depth=max_depth, depth=depth+1,
                                             eran=eran, network=network, relu_layers=relu_layers, verbosity=verbosity, input_nchw=input_nchw)
    else:
        adex_b = None
        result_b = False
    adex = adex_a if adex_a is not None else (adex_b if adex_b is not None else None)
    return result_a and result_b, adex


def verify_network_with_splitting(config, nn, specLB, specUB, constraints, domain, network=None, eran=None,
                                  relu_layers=None, use_parallel_solve=None, max_depth=10, inital_splits=10,
                                  verbosity=2, input_nchw=False):
    def eval_instance(x):
        return evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)

    if use_parallel_solve is None:
        use_parallel_solve = not ("gpu" in domain)

    def init(args):
        global failed_already
        failed_already = args

    multi_bounds = comput_low_dim_init_splits(specLB.copy(), specUB.copy(), eval_instance, initial_splits=inital_splits)
    if len(multi_bounds)<2:
        use_parallel_solve = False

    if use_parallel_solve:
        failed_already = Value('i', False)
        x_adex = None

        try:
            with Pool(processes=min(2*multiprocessing.cpu_count(),len(multi_bounds)), initializer=init, initargs=(failed_already,)) as pool:
                n = len(multi_bounds)
                pool_return = pool.starmap(low_dim_recursive, zip(n * [config], n * [layers()], [x[0] for x in multi_bounds],
                                                                  [x[1] for x in multi_bounds], n * [domain],
                                                                  n * [constraints], n * [max_depth], n * [0],
                                                                  n * [eran], n * [network], n * [relu_layers],
                                                                  n*[verbosity], n*[input_nchw]))
            res = [x[0] for x in pool_return]
            adex = [x[1] for x in pool_return if x[1] is not None]
            for x_adex in adex:  # Should be redundant as only confirmed counterexamples should be returned.
                cex_label, cex_out = eval_instance(x_adex)
                adex_found = not evaluate_cstr(constraints, cex_out)
                if adex_found:
                    break
                else:
                    assert False, "This should not be reachable"

            if all(res):
                verified_flag = True
            else:
                verified_flag = False
        except Exception as ex:
            verified_flag = False
            e = ex
            print("Failed because of an exception ", e)
    else:
        init(Value('i', False))
        verified_flag = True
        for spec_i, (specLB_i, specUB_i) in enumerate(multi_bounds):
            # print(spec_i)
            verified_flag_tmp, x_adex = low_dim_recursive(config, nn, specLB_i, specUB_i, domain, constraints, max_depth=max_depth, depth=0,
                      eran=eran, network=network, relu_layers=relu_layers, verbosity=verbosity, input_nchw=input_nchw)
            if not bool(verified_flag_tmp):
                verified_flag = False
                break
    return verified_flag, [x_adex] if x_adex is not None else None


def verify_network(config, nn, specLB, specUB, constraints, domain, network=None, eran=None, relu_layers=None,
                   find_adex=False, verbosity=2, torch_net=None, input_shape=None, dtype=np.float64, start_time=None,
                   input_nchw=False):
    def eval_instance(x):
        return evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)

    nlb, nub = None, None
    adex = None

    if domain == 'gpupoly' or domain == 'refinegpupoly':
        n_class = len(eval_instance(specLB)[1])
        cstr_matrix, cstr_idxs = cstr_matrix_from_cstr(constraints, n_class=n_class, dtype=network._last_dtype, permit_disjunctive=True)#.astype("float64")
        res = network.evalAffineExpr_withProp(specLB, specUB, cstr_matrix[:, :-1], cstr_matrix[:, -1], back_substitute=network.BACKSUBSTITUTION_WHILE_CONTAINS_ZERO)
        failed_constraint_index = [j for j, cstrs in enumerate(constraints) if not any([res[i, 0] > 0 for i in cstr_idxs[j]])]
        failed_constraints = [[constraints[j][i] for i in sorted(range(len(constraints[j])), key=lambda i: res[cstr_idxs[j][i],0], reverse=True)] for j in failed_constraint_index] # sort or-groups by highest success prob
        failed_constraints = [failed_constraints[j] for j in np.argsort([max([res[i, 0] for i in cstr_idxs[j]]) for j in range(len(failed_constraints))])] # sort and-groups by lowest success prob
        # failed_constraints = sorted(failed_constraints, key=lambda item: )
        is_verified = all([any([res[i, 0] > 0 for i in cstr_idxs[j]]) for j in range(len(constraints))])
        assert (len(failed_constraints)==0) == is_verified

        # failed_constraints = None

        if not is_verified and torch_net is not None and not check_timeout(config, start_time):
            dtype_torch = torch.float64 if dtype == np.float64 else torch.float32
            adex, worst_x = torch_whitebox_attack(torch_net, "cuda", torch.tensor(0.5 * (specLB + specUB).reshape(input_shape), dtype=dtype_torch),
                                         failed_constraints, specLB, specUB, input_nchw, restarts=config.attack_restarts)
            if adex is None and (config.complete or config.partial_milp>0) and worst_x is not None:
                network.eval(worst_x.flatten())
                feas_act = []
                layerno = 0
                for l in range(nn.numlayer):
                    if nn.layertypes[l] in ["FC", "Conv", ]:
                        layerno += 2
                    else:
                        layerno += 1
                    if layerno in relu_layers:
                        feas_acti = feas_act[-1]
                        feas_acti = np.maximum(0, feas_acti)
                    else:
                        bounds = network.evalAffineExpr(layer=layerno)
                        feas_acti = bounds[:, 0]
                    feas_act.append(feas_acti)
                network.evalAffineExpr_withProp(specLB, specUB, cstr_matrix[:, :-1], cstr_matrix[:, -1], back_substitute=1)
            else:
                feas_act = None

        if not is_verified and domain == 'refinegpupoly' and adex is None and not check_timeout(config, start_time):
            nn.specLB = specLB
            nn.specUB = specUB

            constraints_hold, _, nlb, nub, failed_constraints, adex, model_bounds = refine_gpupoly_results(nn, network,
                                                                                                           config,
                                                                                                           relu_layers,
                                                                                                           True,
                                                                                                           adv_labels=-1,
                                                                                                           constraints=constraints,
                                                                                                           K=config.k,
                                                                                                           s=config.s,
                                                                                                           complete=False,
                                                                                                           timeout_final_lp=config.timeout_final_lp,
                                                                                                           timeout_final_milp=config.timeout_final_milp,
                                                                                                           timeout_lp=config.timeout_lp,
                                                                                                           timeout_milp=config.timeout_milp,
                                                                                                           use_milp=config.use_milp,
                                                                                                           partial_milp=config.partial_milp,
                                                                                                           max_milp_neurons=config.max_milp_neurons,
                                                                                                           approx=config.approx_k,
                                                                                                           max_eqn_per_call=config.max_gpu_batch,
                                                                                                           terminate_on_failure=(not config.complete) and (not find_adex),
                                                                                                           eval_instance = eval_instance,
                                                                                                           find_adex=find_adex,start_time=start_time,
                                                                                                           feas_act=feas_act)
            is_verified = is_verified or constraints_hold
    else:
        is_verified = False
        failed_constraints = constraints
        if domain.endswith("poly") and not check_timeout(config, start_time):
            # First do a cheap pass without multi-neuron constraints
            constraints_hold, nn, nlb, nub, failed_constraints, adex, _ = eran.analyze_box(specLB, specUB, "deeppoly",
                                                                                          config.timeout_lp,
                                                                                          config.timeout_milp,
                                                                                          config.use_default_heuristic,
                                                                                          label=-1, prop=-1,
                                                                                          output_constraints=constraints,
                                                                                          K=0, s=0,
                                                                                          timeout_final_lp=config.timeout_final_lp,
                                                                                          timeout_final_milp=config.timeout_final_milp,
                                                                                          use_milp=False,
                                                                                          complete=False,
                                                                                          terminate_on_failure= not config.complete or domain != "refinepoly",
                                                                                          partial_milp=0,
                                                                                          max_milp_neurons=0,
                                                                                          approx_k=0,
                                                                                          eval_instance=eval_instance,
                                                                                          verbosity=verbosity,
                                                                                          start_time=start_time)
            if config.debug:
                print("nlb ", nlb[-1], " nub ", nub[-1], "adv labels ", failed_constraints)
            is_verified = is_verified or constraints_hold

            if adex is not None:
                net_out = evaluate_net(adex, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
                if evaluate_cstr(constraints, net_out):
                    adex = None

        if not is_verified and torch_net is not None and adex is None and not check_timeout(config, start_time):
            dtype_torch = torch.float64 if dtype == np.float64 else torch.float32
            adex, worst_x = torch_whitebox_attack(torch_net, "cpu", torch.tensor(0.5 * (specLB + specUB).reshape(input_shape), dtype=dtype_torch),
                                         failed_constraints, specLB, specUB, input_nchw, restarts=config.attack_restarts)
            if worst_x is not None:
                worst_x = worst_x.flatten()# if input_nchw or len(input_shape)==1 else worst_x.transpose(1, 2, 0).flatten()
                _, _, activations, _, _, _, _ = eran.analyze_box(worst_x, worst_x, 'deepzono', 1, 1, True, label=-1)
            else:
                activations = None


        if (not is_verified) and (not domain.endswith("poly") or "refine" in domain) and adex is None and not check_timeout(config, start_time):
            constraints_hold, nn, nlb, nub, failed_constraints, adex, model_bounds = eran.analyze_box(specLB, specUB,
                                                                                                     domain,
                                                                                                     config.timeout_lp,
                                                                                                     config.timeout_milp,
                                                                                                     config.use_default_heuristic,
                                                                                                     label=-1,
                                                                                                     prop=-1,
                                                                                                     output_constraints=constraints,
                                                                                                     K=config.k,
                                                                                                     s=config.s,
                                                                                                     timeout_final_lp=config.timeout_final_lp,
                                                                                                     timeout_final_milp=config.timeout_final_milp,
                                                                                                     use_milp=config.use_milp,
                                                                                                     complete=False,
                                                                                                     terminate_on_failure=(not config.complete) and (not find_adex),
                                                                                                     partial_milp=config.partial_milp,
                                                                                                     max_milp_neurons=config.max_milp_neurons,
                                                                                                     approx_k=config.approx_k,
                                                                                                     eval_instance = eval_instance,
                                                                                                     verbosity = verbosity,
                                                                                                     start_time=start_time,
                                                                                                     feas_act = activations)

            is_verified = is_verified or constraints_hold

    return is_verified, adex, failed_constraints, nlb, nub, nn


def split_multi_bound(multi_bound, dim=0, d=2):
    if isinstance(d, int):
        di = d
    else:
        di = d[dim]
    new_multi_bound = []
    for specLB, specUB in multi_bound:
        d_lb = specLB[dim]
        d_ub = specUB[dim]
        d_range = d_ub-d_lb
        d_step = d_range/di
        for i in range(di):
            specLB[dim] = d_lb + i*d_step
            specUB[dim] = d_lb + (i+1)*d_step
            new_multi_bound.append((specLB.copy(), specUB.copy()))
    if dim + 1 < len(specUB):
        return split_multi_bound(new_multi_bound, dim=dim+1, d=d)
    else:
        return new_multi_bound


def comput_low_dim_init_splits(specLB, specUB, eval_instance, initial_splits=20):
    if initial_splits <= 1:
        return ([(specLB,specUB)])
    grads = estimate_grads(specLB, specUB, eval_instance)
    # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
    smears = np.multiply(np.abs(grads) + 0.00001, [u - l for u, l in zip(specUB, specLB)])+1e-5

    split_multiple = initial_splits / np.sum(smears)

    num_splits = [int(np.ceil(smear * split_multiple)) for smear in smears]
    assert all([x>0 for x in num_splits])
    return split_multi_bound([(specLB, specUB)], d=num_splits)


def run_analysis():
    config = get_args()

    if config.prepare:
        assert config.netname is not None and config.vnnlib_spec is not None
        config.timeout_complete = 100
        prepare_instance(config=config)
        return None

    if config.res_file is not None:
        with open(config.res_file, "w") as f:
            pass

    if config.instance_file is not None and os.path.exists(config.instance_file):
        with open(config.instance_file, "r") as f:
            lines = f.readlines()
        instances = [[x.strip() for x in line.split(",")] for line in lines]
        instance_dir = os.path.dirname(config.instance_file)
    elif not (config.netname is None or config.vnnlib_spec is None or config.timeout_complete is None):
        instances = [[config.netname, config.vnnlib_spec, config.timeout_complete]]
        instance_dir = None
    else:
        assert False, "No valid spec provided"

    n_unknown = 0
    n_SAT = 0
    n_UNSAT = 0
    cum_time = 0

    for i, instance in enumerate(instances):
        config.n_test = len(instances) if config.n_test == -1 else config.n_test
        if i<config.from_test: continue
        if i-config.from_test >= config.n_test: break

        config.netname = instance[0] if instance_dir is None else os.path.join(instance_dir, instance[0])
        config.vnnlib_spec = instance[1] if instance_dir is None else os.path.join(instance_dir, instance[1])
        config.timeout_complete = float(instance[2])

        time_a = time.time()
        status = run_analysis_instance(config=config)
        cum_time += time.time() - time_a
        if status == "SAT":
            n_SAT += 1
            res_str="violated"
        elif status == "UNSAT":
            n_UNSAT += 1
            res_str = "holds"
        else:
            n_unknown += 1
            res_str = "timeout" if cum_time >= config.timeout_complete else "unknown"
        if config.res_file is not None:
            with open(config.res_file, "a") as f:
                f.write(res_str+"\n")
        print(f"Spec {i} ({status}): {i+1-config.from_test}/{len(instances)-config.from_test}; SAT {n_SAT}; UNSAT {n_UNSAT}; unknown {n_unknown};",
              f"time: {time.time() - time_a:.3f}; {cum_time / (i+1-config.from_test):.3f}; {cum_time:.3f}")


def prepare_instance(config=None):
    if config is None:
        config = get_args(verbosity=0)

    if config.netname.endswith(".gz"):
        if not os.path.exists(config.netname[:-3]):
            block_size = 65536
            with gzip.open(config.netname, 'rb') as s_file, \
                    open(config.netname[:-3], 'wb') as d_file:
                shutil.copyfileobj(s_file, d_file, block_size)
        config.netname = config.netname[:-3]

    filename, file_extension = os.path.splitext(config.netname)

    is_onnx = file_extension == ".onnx"
    assert is_onnx, "file extension not supported in ERAN light"
    pkl_file_net = f"{filename}.pkl"
    netname = config.netname
    domain = config.domain

    ### Prepare analysis, translating the model
    model, is_conv = read_onnx_net(netname)

    nn = layers()
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        translator = ONNXTranslator(model, True)
        operations, resources, orig_input_shape = translator.translate()
        optimizer = Optimizer(operations, resources)
        network, relu_layers, num_gpu_layers, _, nn = optimizer.get_gpupoly(nn)
        if pkl_file_net is not None:
            for i in range(len(resources)):
                for j in ["deeppoly", "deepzono"]:
                    res = ()
                    for k in range(len(resources[i][j])):
                        if isinstance(resources[i][j][k], google.protobuf.pyext._message.RepeatedScalarContainer):
                            res += (list(resources[i][j][k]),)
                        else:
                            res += (resources[i][j][k],)
                    resources[i][j] = res
            pkl.dump((resources, operations), open(pkl_file_net, "wb"))
    else:
        eran = ERAN(model, is_onnx=True, pkl_file=pkl_file_net)
        orig_input_shape = eran.orig_input_shape

    os.sched_setaffinity(0, cpu_affinity)
    mean = config.mean
    std = config.std
    nchw = "gpu" in domain

    boxes, constraint_set, _, _, is_nchw, input_shape = vnn_lib_data_loader(config.vnnlib_spec, dtype=DTYPE, index_is_nchw=config.index_is_nchw, output_is_nchw=nchw)
    pkl_file_prop = f"{config.vnnlib_spec[:-7]}.pkl"
    pkl.dump((boxes, constraint_set, mean, std, is_nchw, input_shape), open(pkl_file_prop, "wb"))

    orig_sample, eps = translate_box_to_sample(boxes, equal_limits=True)
    check_translation(orig_sample[0], input_shape, index_nchw=config.index_is_nchw, is_nchw=nchw, onnx_model=model, onnx_input_shape= orig_input_shape,
                      evalute_net=lambda x: evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None))

    if os.path.exists(netname[:-5]+".pynet"):
        os.remove(netname[:-5]+".pynet")

    torch_net = convert_net(netname, [orig_sample[0], boxes[0][0], boxes[0][1]], input_shape, nchw,
                            lambda x: evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None))
    torch.save(torch_net, netname[:-5] + ".pynet")

    return config

def run_analysis_instance(config=None):
    load_from_prepare = True
    if config is None:
        config = get_args()

    if config.netname.endswith(".gz"):
        if not os.path.exists(config.netname[:-3]):
            block_size = 65536
            with gzip.open(config.netname, 'rb') as s_file, \
                    open(config.netname[:-3], 'wb') as d_file:
                shutil.copyfileobj(s_file, d_file, block_size)
        config.netname = config.netname[:-3]

    filename, file_extension = os.path.splitext(config.netname)

    is_onnx = file_extension == ".onnx"
    assert is_onnx, "file extension not supported in ERAN light"
    pkl_file_net = f"{filename}.pkl"
    pkl_file_prop = f"{config.vnnlib_spec[:-7]}.pkl"

    netname = config.netname
    domain = config.domain

    ### Prepare analysis, translating the model
    model, is_conv = read_onnx_net(netname)

    nn = layers()
    if domain == 'gpupoly' or domain == 'refinegpupoly':
        if load_from_prepare and os.path.exists(pkl_file_net):
            with open(pkl_file_net, 'rb') as f:
                (resources, operations, orig_input_shape) = pkl.load(f)
            print(f"Loaded resources successfully")
        else:
            translator = ONNXTranslator(model, True)
            operations, resources, orig_input_shape = translator.translate()
        optimizer = Optimizer(operations, resources)
        network, relu_layers, num_gpu_layers, _, nn = optimizer.get_gpupoly(nn)
    else:
        if load_from_prepare and os.path.exists(pkl_file_net):
            with open(pkl_file_net, 'rb') as f:
                (resources, operations, orig_input_shape) = pkl.load(f)
            print(f"Loaded resources successfully")
        else:
            resources, operations, orig_input_shape = None, None, None
        eran = ERAN(model, is_onnx=True, resources=resources, operations=operations, orig_input_shape=orig_input_shape)
        orig_input_shape = eran.orig_input_shape

    mean = config.mean
    std = config.std

    os.sched_setaffinity(0, cpu_affinity)
    ### Initialize counters
    correctly_classified_images = 0
    verified_images = 0
    unsafe_images = 0
    cum_time = 0
    nchw = "gpu" in domain

    if load_from_prepare and os.path.exists(pkl_file_prop):
        with open(pkl_file_prop, 'rb') as f:
            (boxes, constraint_set, mean, std, is_nchw, input_shape) = pkl.load(f)
        print(f"Loaded spec successfully")
    else:
        boxes, constraint_set, _, _, is_nchw, input_shape = vnn_lib_data_loader(config.vnnlib_spec, dtype=DTYPE, index_is_nchw=config.index_is_nchw, output_is_nchw=nchw)
    # orig_sample, eps = translate_box_to_sample(boxes, equal_limits=len(input_shape)>2)
    check_translation(boxes[0][0], input_shape, index_nchw=config.index_is_nchw, is_nchw=nchw, onnx_model=model, onnx_input_shape= orig_input_shape,
                      evalute_net=lambda x: evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None),
                      mean=mean, std=std)
    boxes = [(normalize(specLB,mean,std,nchw,input_shape),normalize(specUB,mean,std,nchw,input_shape)) for specLB, specUB in boxes]
    orig_sample, eps = translate_box_to_sample(boxes, equal_limits=len(input_shape)>2)

    if os.path.exists(netname[:-5]+".pynet"):
        torch_net = torch.load(netname[:-5] + ".pynet")
        print(f"Loaded pynet successfully")
    else:
        torch_net = convert_net(netname, [orig_sample[0], boxes[0][0], boxes[0][1]], input_shape, nchw,
                                lambda x: evaluate_net(x, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None))
        torch.save(torch_net, netname[:-5] + ".pynet")
    torch_net = torch_net.to("cuda" if "gpu" in domain else "cpu").eval()


    # fb_model, fb_session = onnx_2_tf(netname, mean, std, bounds=(0, 1))

    spec_sat = False
    spec_unsat = True
    global start_time
    start_time = time.time()

    for i, (x, y) in enumerate(zip(boxes, constraint_set)):
        # label = extract_label_from_cstr(y)
        y_1 = evaluate_net(x[0], domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
        y_2 = evaluate_net(x[1], domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
        y_3 = evaluate_net(orig_sample, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
        is_correctly_classified = evaluate_cstr(y, np.stack([y_1[1], y_2[1], y_3[1]], axis=0)).all()
        if config.debug:
            print(f"concrete outputs: {y_3[1]}")
        adex_found = not is_correctly_classified
        is_verified = False

        ## start certification
        if is_correctly_classified:
            # Only attempt certification for correctly classified samples
            correctly_classified_images += 1

            specLB = x[0]
            specUB = x[1]
            constraints = y

            adex = None

            if adex is None:
                is_verified, adex, failed_constraints, nlb, nub, nn = verify_network(config, nn, specLB, specUB, constraints,
                                              domain, network =  network if "gpu" in domain else None,
                                              eran = eran if "gpu" not in domain else None,
                                              relu_layers=relu_layers if "gpu" in domain else None, find_adex=True,
                                              torch_net=torch_net, input_shape=input_shape, start_time=start_time, input_nchw=nchw)
            if is_verified:
                print("img", i, f"Verified UNSAT for constraints {constraints}")
                verified_images += 1
            else:
                if adex is not None:
                    for x_i in adex:
                        # Check if returned solution is an adversarial example
                        cex_label, cex_out = evaluate_net(x_i, domain, network if "gpu" in domain else None, eran if "gpu" not in domain else None)
                        adex_found = not evaluate_cstr(constraints, cex_out)
                        # adex_found = cex_label != label
                        if adex_found:
                            print(f"img {i} Verified SAT, with adversarial example with output {cex_out} for constraints {constraints}")
                            # denormalize(np.array(x_i), mean, std, is_nchw, input_shape)
                            unsafe_images += 1
                            break
                if not adex_found:
                    if len(specLB) < 10:
                        # config.timeout_complete = 500
                        is_verified, adex = verify_network_with_splitting(config, nn, specLB, specUB, constraints,
                                              domain, network =  network if "gpu" in domain else None,
                                              eran = eran if "gpu" not in domain else None,
                                              relu_layers=relu_layers if "gpu" in domain else None,
                                              max_depth=config.max_split_depth, inital_splits=config.initial_splits,
                                              verbosity=1, input_nchw=nchw)
                    elif config.complete and (failed_constraints is not None):
                        # Run complete verification on uncertified constraints
                        is_verified, adex, adv_val = verify_network_with_milp(config, nn, specLB, specUB, nlb, nub,
                                                                              failed_constraints,
                                                                              gpu_model="gpu" in domain,
                                                                              find_adex=True, is_nchw=nchw, start_time=start_time)
                    if is_verified:
                        print("Box", i, "Verified UNSAT for constraints", constraints)
                        verified_images += 1
                    else:
                        if adex is not None:
                            for x_i in adex:
                                # Check if returned solution is an adversarial example
                                cex_label, cex_out = evaluate_net(x_i, domain, network if "gpu" in domain else None,
                                                                  eran if "gpu" not in domain else None)
                                adex_found = not evaluate_cstr(constraints, cex_out)
                                # adex_found = cex_label != label
                                if adex_found:
                                    print(
                                        f"Box {i} Verified SAT, with adversarial example with output {cex_out} for constraints {constraints}")
                                    # denormalize(np.array(x_i), mean, std, is_nchw, input_shape)
                                    unsafe_images += 1
                                    break
                        if not adex_found:
                            print(f"Box {i} Failed, without a adversarial example")
                        else:
                            print(f"Box {i} Failed")
            end = time.time()
            cum_time += end - start_time  # only count samples where we did try to certify
        else:
            end = time.time()
            print(f"Box {i} SAT")

        spec_unsat = spec_unsat and is_verified
        spec_sat = adex_found

        print(f"progress: {int(1 + i)}/{len(boxes)}")
    status = "SAT" if spec_sat else ("UNSAT" if spec_unsat else "UNKNOWN")
    return status


def check_translation(image, input_shape, index_nchw, is_nchw, onnx_model, onnx_input_shape, evalute_net, mean=None, std=None):
    if mean is not None or std is not None:
        eran_image = normalize(image, mean, std, is_nchw, input_shape)
    else:
        eran_image = image
    net_out_eran = evalute_net(eran_image)[1]

    onnx.checker.check_model(onnx_model)
    base = rt.prepare(onnx_model, 'CPU')

    input_dim = len(input_shape)
    if input_dim in [3,1]:
        input_shape = [1,] + list(input_shape)
        input_dim += 1

    nhwc_input_shape = input_shape[0:1] + input_shape[2:4] + input_shape[1:2]


    if is_nchw and input_dim==4:
        image = image.reshape(input_shape).transpose(0, 2, 3, 1)

    if index_nchw and input_dim==4: # image is always in nhwc
        image = image.flatten().reshape(nhwc_input_shape).transpose(0, 3, 1, 2)
    elif input_dim == 4:
        image = image.flatten().reshape(nhwc_input_shape)

    try:
        net_out_onnx = base.run(image.astype(np.float32).reshape(onnx_input_shape))
    except Exception as e1:
        try:
            net_out_onnx = base.run(image.astype(np.float64).reshape(onnx_input_shape))
        except Exception as e2:
            assert False, f"Failed to run onnx model with two exceptions: \n{e1}\n\n{e2}"

    assert np.isclose(net_out_onnx, net_out_eran, rtol=1e-3, atol=1e-6).all(), net_out_eran


def convert_net(netname, inputs, input_shape, nchw, evaluate_net):
    model, is_conv = read_onnx_net(netname)
    translator = ONNXTranslator(model, True)
    operations, resources, _ = translator.translate()

    if input_shape is None:
        input_shape = inputs.shape
    else:
        input_shape = [1,] + list(input_shape)

    nhwc_input_shape = input_shape[0:1] + input_shape[2:4] + input_shape[1:2]

    nhwc_flag = True
    nchw_flag = True

    means, stds = None, None
    net_nhwc, layers_nhwc = create_torch_net_new(operations, resources, means, stds, nhwc_input_shape, False)
    net_nchw, layers_nchw = create_torch_net_new(operations, resources, means, stds, nhwc_input_shape, True)

    for i, test in enumerate(inputs):
        if i>2:
            break
        image = np.float64(test)
        net_out = evaluate_net(image)[1]

        if len(input_shape)>2:
            if nchw:
                image = image.reshape(input_shape).transpose(0, 2, 3, 1).flatten() # converts all inputs to nhwc format
            torch_image = torch.tensor(image.reshape(nhwc_input_shape), dtype=torch.float64).permute((0, 3, 1, 2))
        else:
            torch_image = torch.tensor(image.reshape(nhwc_input_shape), dtype=torch.float64)

        if nhwc_flag:
            torch_out_nhwc = net_nhwc(torch_image).detach().cpu().numpy()
            if not np.isclose(net_out, torch_out_nhwc, rtol=1e-3, atol=5e-4).all():
                nhwc_flag = False

        if nchw_flag:
            torch_out_nchw = net_nchw(torch_image).detach().cpu().numpy()
            if not np.isclose(net_out, torch_out_nchw, rtol=1e-3, atol=5e-4).all():
                nchw_flag = False

    if nhwc_flag and nchw_flag:
        print("Both nchw and nhwc lead to correct outputs")
        return net_nchw
    elif nchw_flag:
        print("Nchw leads to correct outputs")
        return net_nchw
    elif nhwc_flag:
        print("Nhwc leads to correct outputs")
        return net_nhwc
    else:
        assert False, "Both nchw and nhwc lead to mismatched outputs"

if __name__ == '__main__':
    run_analysis()