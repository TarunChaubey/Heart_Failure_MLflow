# install init_setup.sh
bash init_setup.sh

# Run MLProjec file

#case 1 - when have to install setup from start
mlflow run . 

#only run program in prepared enviroment
mlflow run . --no-conda