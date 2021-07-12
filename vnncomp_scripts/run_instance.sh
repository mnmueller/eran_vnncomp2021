#!/bin/bash
# prepare_instance.sh script for VNNCOMP for ERAN: # four arguments, first is "v1", second is a benchmark category identifier string, third is path to the .onnx file and fourth is path to .vnnlib file
# Stanley Bak, Feb 2021

TOOL_NAME=eran
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

CATEGORY=$2
ONNX_FILE=$3
VNNLIB_FILE=$4
RESULTS_FILE=$5
TIMEOUT=$6

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
eran_file="$SCRIPT_DIR/../tf_verify/eran_light.py"

echo "Running benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE', vnnlib file '$VNNLIB_FILE', results file $RESULTS_FILE, and timeout $TIMEOUT"

if [[ ! $CONDA_DEFAULT_ENV == "ERAN" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate ERAN
  echo "activated conda environment"
fi


case $CATEGORY in
  acasxu)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain deeppoly --timeout_final_milp 5 --complete True --max_split_depth 8 --initial_splits 10 --attack_restarts 1
  ;;
  cifar10_resnet)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 50 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 50 --timeout_final_milp 300 --attack_restarts 1
  ;;
  cifar2020)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 100 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 100 --timeout_final_milp 100 --attack_restarts 5
  ;;
  eran)
    if [[ "$ONNX_FILE" == *"SIGMOID"* ]]; then
      python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinepoly --sparse_n 10 --k 3 --s -2 --refine_neurons --timeout_milp 10 --timeout_lp 10 --n_milp_refine 2 --attack_restarts 2
    else
      python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinepoly --sparse_n 50 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 100 --refine_neurons --timeout_milp 10 --timeout_lp 10 --n_milp_refine 2 --attack_restarts 2
    fi
  ;;
  marabou-cifar10)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 20 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 50 --timeout_final_milp 120 --index_is_nchw False --attack_restarts 10
  ;;
  mnistfc)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 100 --k 3 --s -2 --partial_milp 4 --max_milp_neurons 50 --timeout_final_milp 200 --refine_neurons --n_milp_refine 2 --timeout_milp 5 --timeout_lp 2  --attack_restarts 10
  ;;
  nn4sys)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain gpupoly --timeout_final_milp 20 --complete True --max_split_depth 4 --initial_splits 1 --attack_restarts 1
  ;;
  oval21)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 100 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 30 --timeout_final_milp 200 --attack_restarts 10
  ;;
  verivital)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 100 --k 3 --s -2 --partial_milp 1 --max_milp_neurons 100 --timeout_final_milp 200 --complete True --attack_restarts 2
  ;;
  test)
    python3 "$SCRIPT_DIR/../tf_verify/test_gurobi.py"
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 100 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 100 --timeout_final_milp 200 --attack_restarts 10
  ;;
  *)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --timeout_complete $TIMEOUT --res_file $RESULTS_FILE --domain refinegpupoly --sparse_n 100 --k 3 --s -2 --partial_milp 2 --max_milp_neurons 100 --timeout_final_milp 200 --attack_restarts 10
  ;;
esac
# script returns a 0 exit code if successful. If you want to skip a benchmark category you can return non-zero.
#if [ "$CATEGORY" = "test" -o "$CATEGORY" == "acasxu" ]
#then
#	exit 0
#fi

exit 0
