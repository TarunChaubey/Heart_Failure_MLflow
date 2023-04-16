conda create --prefix ./env python=3.7 -y
# source C:/Users/Asus/anaconda3/etc/profile.d/conda.sh # use your username instead of Asus
# source activate ./env
conda activate ./env
pip install -r requirements.txt
conda env export > conda.yaml