## Setup Instructions
1) Clone ERAN repository with
    ```
    git clone https://github.com/mnmueller/eran_vnncomp2021.git
    cd eran_vnncomp2021
    ```
2) Create and activate a conda environment:
   ```
   conda create --name ERAN python=3.6 -y
   conda activate ERAN
   ```
3) Run the install script (wait until system updates are done (`ps -aux | grep apt` shows only the grep command)):
   ```
   bash ./vnncomp_scripts/install_tool.sh v1
   ```
4) Acquire a gurobi license and install:
   ```
   cd gurobi903/linux64/bin
   ./grbgetkey <license-key>
   ```

## Run Info
ERAN should be run on a GPU instance