#!/bin/bash

set -e

if ! [ $(id -u) = 0 ]; then
   echo "The script needs to be run as root."
   exit 1
fi

if [ $SUDO_USER ]; then
    real_user=$SUDO_USER
else
    real_user=$(whoami)
fi

apt-get install m4 -y
apt-get install build-essential -y
apt-get install autoconf -y
apt-get install libtool -y
apt-get install texlive-latex-base -y 

if test -L /usr/local/cuda; then
  rm /usr/local/cuda
fi
ln -s /usr/local/cuda-11.0/ /usr/local/cuda

# AWS AMI instances have sufficiently recent cmake version installed already
#if ! command -v cmake &> /dev/null
#then
#  CMAKE_DIR=$(which cmake)
#else
#  CMAKE_DIR=/usr/bin/cmake
#fi
#wget https://github.com/Kitware/CMake/releases/download/v3.19.7/cmake-3.19.7-Linux-x86_64.sh
#bash ./cmake-3.19.7-Linux-x86_64.sh
#if test -f "$CMAKE_DIR" | test -L "$CMAKE_DIR"; then
#  rm "$CMAKE_DIR"
#fi
#ln -s "$(pwd)/cmake-3.19.7-Linux-x86_64/bin/cmake" "$CMAKE_DIR"


has_cuda=1

wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx
make
make install
cd ..
rm gmp-6.1.2.tar.xz

wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz
tar -xvf mpfr-4.1.0.tar.xz
cd mpfr-4.1.0
./configure
make
make install
cd ..
rm mpfr-4.1.0.tar.xz

wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz
tar zxf cddlib-0.94m.tar.gz
cd cddlib-0.94m
./configure
make
make install
cd ..

wget https://packages.gurobi.com/9.0/gurobi9.0.3_linux64.tar.gz
sudo -u $real_user tar -xvf gurobi9.0.3_linux64.tar.gz
cd gurobi903/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
export GUROBI_HOME="$(pwd)"
cp lib/libgurobi90.so /usr/local/lib
python3 setup.py install
cd ../../
rm gurobi9.0.3_linux64.tar.gz

export PATH="${PATH}:/usr/lib:${GUROBI_HOME}/bin:/usr/local/cuda/bin"
export CPATH="${CPATH}:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib:${GUROBI_HOME}/lib

sudo -u $real_user git clone https://github.com/eth-sri/ELINA.git
cd ELINA
sudo -u $real_user git checkout 2519e91d4d4d2b0b3e9163a53618c3d04892eb83
if test "$has_cuda" -eq 1
then
    ./configure -use-cuda -use-deeppoly -use-gurobi -use-fconv
    cd ./gpupoly/
    cmake .
    cd ..
else
    ./configure -use-deeppoly -use-gurobi -use-fconv
fi
make
make install
cd ..
chmod 777 ./vnncomp_scripts/*

git clone https://github.com/eth-sri/deepg.git
cd deepg/code
mkdir build
make shared_object
cp ./build/libgeometric.so /usr/lib
cd ../..

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:/usr/lib

ldconfig