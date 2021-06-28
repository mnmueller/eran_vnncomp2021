## Setup Instructions
1) Clone ERAN repository with
    ```
    git clone https://github.com/mnmueller/eran_vnncomp2021.git
    cd eran_vnncomp2021
    ```
2) Run the install script:
   ```
   bash ./vnncomp_scripts/install_tool.sh v1
   ```
3) Acquire a gurobi license and install:
   ```
   cd gurobi903/linux64/bin
   ./grbgetkey <license-key>
   ```

## Run Info
ERAN should be run on a GPU instance