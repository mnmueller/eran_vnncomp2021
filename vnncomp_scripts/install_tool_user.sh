#!/bin/bash

cd gurobi903/linux64/
if [[ $CONDA_DEFAULT_ENV == "" ]]; then
  python3 setup.py install
else
  ~/anaconda3/envs/$CONDA_DEFAULT_ENV/bin/python3 setup.py install
fi


cd bin
./grbgetkey 878b2d60-d7db-11eb-be80-0242ac120002 < ../../
cd ../../../

if [[ $CONDA_DEFAULT_ENV == "" ]]; then
  python3 -m pip install -r requirements.txt
else
  ~/anaconda3/envs/$CONDA_DEFAULT_ENV/bin/python3 -m pip install -r requirements.txt
fi
