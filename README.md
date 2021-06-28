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
3) Check that gurobi license was obtained successfully by calling `python3` and running:
    ```
    import gurobipy
    gurobipy.Model()
    exit()
    ```
   If an error is returned, acquire a gurobi license