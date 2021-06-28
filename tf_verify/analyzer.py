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


from elina_abstract0 import *
from elina_manager import *
from deeppoly_nodes import *
from deepzono_nodes import *
from ai_milp import evaluate_models
from functools import reduce
from ai_milp import milp_callback
from utils import check_timeleft
import gc

class layers:
    def __init__(self):
        self.layertypes = []
        self.weights = []
        self.biases = []
        self.filters = []
        self.numfilters = []
        self.filter_size = [] 
        self.input_shape = []
        self.strides = []
        self.padding = []
        self.out_shapes = []
        self.pool_size = []
        self.numlayer = 0
        self.ffn_counter = 0
        self.conv_counter = 0
        self.residual_counter = 0
        self.pad_counter = 0
        self.pool_counter = 0
        self.concat_counter = 0
        self.tile_counter = 0
        self.activation_counter = 0
        self.specLB = []
        self.specUB = []
        self.original = []
        self.zonotope = []
        self.predecessors = []
        self.lastlayer = None
        self.last_weights = None
        self.label = -1
        self.prop = -1

    def calc_layerno(self):
        return self.ffn_counter + self.conv_counter + self.residual_counter + self.pool_counter + self.activation_counter + self.concat_counter + self.tile_counter +self.pad_counter

    def is_ffn(self):
        return not any(x in ['Conv2D', 'Conv2DNoReLU', 'Resadd', 'Resaddnorelu'] for x in self.layertypes)

    def set_last_weights(self, constraints):
        # TODO this should be replaced
        length = 0.0       
        last_weights = [0 for weights in self.weights[-1][0]]
        for or_list in constraints:
            for (i, j, k) in or_list:
                if j == -1:
                    last_weights = [l + w_i + float(k) for l, w_i in zip(last_weights, self.weights[-1][i])]
                elif i == -1:
                    last_weights = [l + w_i + float(k) for l, w_i in zip(last_weights, self.weights[-1][i])]
                else:
                    last_weights = [l + w_i + w_j + float(k) for l, w_i, w_j in zip(last_weights, self.weights[-1][i], self.weights[-1][j])]
                length += 1
        self.last_weights = [w/length for w in last_weights]


    def back_propagate_gradient(self, nlb, nub):
        #assert self.is_ffn(), 'only supported for FFN'
        # TODO this should be replaced

        grad_lower = self.last_weights.copy()
        grad_upper = self.last_weights.copy()
        last_layer_size = len(grad_lower)
        for layer in range(len(self.weights)-2, -1, -1):
            weights = self.weights[layer]
            lb = nlb[layer]
            ub = nub[layer]
            layer_size = len(weights[0])
            grad_l = [0] * layer_size
            grad_u = [0] * layer_size

            for j in range(last_layer_size):

                if ub[j] <= 0:
                    grad_lower[j], grad_upper[j] = 0, 0

                elif lb[j] <= 0:
                    grad_upper[j] = grad_upper[j] if grad_upper[j] > 0 else 0
                    grad_lower[j] = grad_lower[j] if grad_lower[j] < 0 else 0

                for i in range(layer_size):
                    if weights[j][i] >= 0:
                        grad_l[i] += weights[j][i] * grad_lower[j]
                        grad_u[i] += weights[j][i] * grad_upper[j]
                    else:
                        grad_l[i] += weights[j][i] * grad_upper[j]
                        grad_u[i] += weights[j][i] * grad_lower[j]
            last_layer_size = layer_size
            grad_lower = grad_l
            grad_upper = grad_u
        return grad_lower, grad_upper




class Analyzer:
    def __init__(self, ir_list, nn, domain, timeout_lp, timeout_milp, output_constraints, use_default_heuristic, label=-1,
                 prop=-1, testing=False, K=3, s=-2, timeout_final_lp=100, timeout_final_milp=100, use_milp=False,
                 complete=False, partial_milp=False, max_milp_neurons=30, approx_k=True, eval_instance=None, feas_act=None):
        """
        Arguments
        ---------
        ir_list: list
            list of Node-Objects (e.g. from DeepzonoNodes), first one must create an abstract element
        domain: str
            either 'deepzono', 'refinezono' or 'deeppoly'
        """
        self.ir_list = ir_list
        self.is_greater = is_greater_zono
        self.refine = False
        if domain == 'deeppoly' or domain == 'refinepoly':
            self.man = fppoly_manager_alloc()
            self.is_greater = is_greater
        elif domain == 'deepzono' or domain == 'refinezono':
            self.man = zonoml_manager_alloc()
            self.is_greater = is_greater_zono
        if domain == 'refinezono' or domain == 'refinepoly':
            self.refine = True
        self.domain = domain
        self.nn = nn
        self.timeout_lp = timeout_lp
        self.timeout_milp = timeout_milp
        self.timeout_final_lp = timeout_final_lp
        self.timeout_final_milp = timeout_final_milp
        self.use_milp = use_milp
        self.output_constraints = output_constraints
        self.use_default_heuristic = use_default_heuristic
        self.testing = testing
        self.relu_groups = []
        self.label = label
        self.prop = prop
        self.complete = complete
        self.K=K
        self.s=s
        self.partial_milp=partial_milp
        self.max_milp_neurons=max_milp_neurons
        self.approx_k = approx_k
        self.eval_instance = eval_instance
        self.feas_act = feas_act

    def __del__(self):
        elina_manager_free(self.man)
        
    
    def get_abstract0(self, start_time=None):
        """
        processes self.ir_list and returns the resulting abstract element
        """
        element = self.ir_list[0].transformer(self.man)
        nlb = []
        nub = []
        testing_nlb = []
        testing_nub = []
        for i in range(1, len(self.ir_list)):
            if type(self.ir_list[i]) in [DeeppolyReluNode, DeeppolySigmoidNode, DeeppolyTanhNode, DeepzonoRelu, DeepzonoSigmoid, DeepzonoTanh]:
                element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub,
                                                                  self.relu_groups, 'refine' in self.domain,
                                                                  self.timeout_lp, self.timeout_milp,
                                                                  self.use_default_heuristic, self.testing,
                                                                  K=self.K, s=self.s, use_milp=self.use_milp,
                                                                  approx=self.approx_k, start_time=start_time)
            else:
                element_test_bounds = self.ir_list[i].transformer(self.nn, self.man, element, nlb, nub,
                                                                  self.relu_groups, 'refine' in self.domain,
                                                                  self.timeout_lp, self.timeout_milp,
                                                                  self.use_default_heuristic, self.testing)

            if self.testing and isinstance(element_test_bounds, tuple):
                element, test_lb, test_ub = element_test_bounds
                testing_nlb.append(test_lb)
                testing_nub.append(test_ub)
            else:
                element = element_test_bounds
        if self.domain in ["refinezono", "refinepoly"]:
            gc.collect()
        if self.testing:
            return element, testing_nlb, testing_nub
        return element, nlb, nub
    
    
    def analyze(self,terminate_on_failure=True, verbosity=2, start_time=None):
        """
        analyses the network with the given input
        
        Returns
        -------
        output: int
            index of the dominant class. If no class dominates then returns -1
        """
        element, nlb, nub = self.get_abstract0(start_time)
        
        # if self.domain == "deeppoly" or self.domain == "refinepoly":
        #     linexprarray = backsubstituted_expr_for_layer(self.man, element, 1, True)
        #     for neuron in range(1):
        #         print("******EXPR*****")
        #         elina_linexpr0_print(linexprarray[neuron],None)
        #         print()
        # output_size = 0
        if self.domain == 'deepzono' or self.domain == 'refinezono':
            output_size = self.ir_list[-1].output_length
        else:
            output_size = self.ir_list[-1].output_length#reduce(lambda x,y: x*y, self.ir_list[-1].bias.shape, 1)
        
        dominant_class = -1
        if self.domain=='refinepoly' and not check_timeout(config, start_time, 0.1):
            counter, var_list, model = create_model(self.nn, self.nn.specLB, self.nn.specUB, nlb, nub, self.relu_groups,
                                                    self.nn.numlayer, is_nchw=False, use_milp=False)
            model.setParam(GRB.Param.TimeLimit, check_timeleft(config,start_time,config.timeout_final_lp))
            model.setParam(GRB.Param.Cutoff, 0.01)

            if self.partial_milp != 0:
                counter_partial_milp, var_list_partial_milp, model_partial_milp = create_model(self.nn, self.nn.specLB,
                                                                                               self.nn.specUB, nlb, nub,
                                                                                               self.relu_groups,
                                                                                               self.nn.numlayer,
                                                                                               is_nchw=False,
                                                                                               use_milp=False,
                                                                                               partial_milp=self.partial_milp,
                                                                                               max_milp_neurons=self.max_milp_neurons,
                                                                                               feas_act=self.feas_act)
                model_partial_milp.setParam(GRB.Param.TimeLimit, check_timeleft(config, start_time, config.timeout_final_milp))
                model_partial_milp.setParam(GRB.Param.Cutoff, 0.01)
            else:
                model_partial_milp = None
                var_list_partial_milp = None
                counter_partial_milp = None



            num_var = len(var_list)
            output_size = num_var - counter

        constraint_dict = {}

        if self.output_constraints is None:
            if config.regression:
                constraints = []
                if self.label == -1:
                    assert False, "For a regression taks a target needs to be provided for certificaiton"
                else:
                    constraints.append([(0, -1, self.label - config.epsilon_y)])
                    constraints.append([(-1, 0, self.label + config.epsilon_y)])
                    constraint_dict[self.label] = constraints
            else:
                if self.label == -1:
                    # Check all labels if non provided
                    candidate_labels = list(range(output_size))
                else:
                    candidate_labels = [self.label]

                if self.prop == -1:
                    # Check against all labels if non provided
                    adv_labels = list(range(output_size))
                else:
                    adv_labels = [self.prop]

                for label in candidate_labels:
                    # Translate label checks in constraints
                    constraints = []
                    for adv_label in adv_labels:
                        if label == adv_label:
                            continue
                        if nlb[-1][label] > nub[-1][adv_label]:
                            continue
                        constraints.append([(label, adv_label, 0)])
                    constraint_dict[label] = constraints
        else:
            constraint_dict[True] = self.output_constraints

        label_failed = []
        adex_list = []
        model_bounds = None

        for constrain_key in constraint_dict.keys():
            # AND
            and_result = True
            failed_constraints = []
            for or_list in constraint_dict[constrain_key]:
                # if check_timeout(config,start_time,0.001):
                #     continue
                # OR
                or_result = False
                ### First do a pass with a cheap method
                adex_list_or = []
                for is_greater_tuple in or_list:
                    if is_greater_tuple[1] == -1:
                        if nlb[-1][is_greater_tuple[0]] > float(is_greater_tuple[2]):
                            or_result = True
                            break
                    elif is_greater_tuple[0] == -1:
                        if float(is_greater_tuple[2]) > nub[-1][is_greater_tuple[1]]:
                            or_result = True
                            break
                    else:
                        if nlb[-1][is_greater_tuple[0]] > nub[-1][is_greater_tuple[1]]:
                            or_result = True
                            break
                        if self.domain == 'deepzono' or self.domain == 'refinezono':
                            if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1]):
                                or_result = True
                                break
                        else:
                            if self.is_greater(self.man, element, is_greater_tuple[0], is_greater_tuple[1],
                                               self.use_default_heuristic):
                                or_result = True
                                break
                if not or_result:
                    and_result = False
                    failed_constraints.append(or_list)
            if self.domain == 'refinepoly' and not and_result and not check_timeout(config,start_time):
                and_result, failed_constraints, adex_list_and, model_bounds = evaluate_models(model, var_list, counter,
                                                                                              len(self.nn.specLB),
                                                                                              failed_constraints,
                                                                                              terminate_on_failure,
                                                                                              model_partial_milp,
                                                                                              var_list_partial_milp,
                                                                                              counter_partial_milp,
                                                                                              eval_instance=self.eval_instance,
                                                                                              verbosity=verbosity, start_time=start_time)

                if len(constraint_dict.keys()) == 1:
                    label_failed += [x[0][1] for x in failed_constraints]
                    adex_list += adex_list_and
            elif len(constraint_dict.keys()) == 1 and len(failed_constraints)>0 and all([len(x)==1 for x in failed_constraints]):
                # Do not return failed labels when testing for multiple "true" labels
                # Only return failed labels, when all or clauses have only one element
                label_failed += [x[0][1] for x in failed_constraints]

            if and_result:
                dominant_class = constrain_key
                break
        if dominant_class == -1 and isinstance(constrain_key,bool):
            dominant_class = False

        failed_constraints = failed_constraints if len(failed_constraints) > 0 else None
        adex_list = adex_list if len(adex_list) > 0 else None

        elina_abstract0_free(self.man, element)
        return dominant_class, nlb, nub, failed_constraints, adex_list, model_bounds