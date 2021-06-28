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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
eran_file="$SCRIPT_DIR/../tf_verify/eran_light.py"

echo "Preparing $TOOL_NAME for benchmark instance in category '$CATEGORY' with onnx file '$ONNX_FILE' and vnnlib file '$VNNLIB_FILE'"

# kill any zombie processes
killall -q python3

case $CATEGORY in
  acasxu)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain deeppoly --prepare
  ;;
  cifar10_resnet)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
  cifar2020)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
  eran)
    if [[ "$ONNX_FILE" == *"SIGMOID"* ]]; then
      python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinepoly --prepare
    else
      python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinepoly --prepare
    fi
  ;;
  marabou-cifar10)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
  mnistfc)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
  nn4sys)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain gpupoly --prepare
  ;;
  oval21)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
  verivital)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
  *)
    python3 $eran_file  --netname $ONNX_FILE --vnnlib_spec $VNNLIB_FILE --domain refinegpupoly --prepare
  ;;
esac

exit 0
