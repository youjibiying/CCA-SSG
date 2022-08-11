# Cora
python main.py --dataname cora --epochs 50 --lambd 1e-3 --dfr 0.1 --der 0.4 --lr2 1e-2 --wd2 1e-4

# Citeseer
python main.py --dataname citeseer --epochs 20 --n_layers 1 --lambd 5e-4 --dfr 0.0 --der 0.4 --lr2 1e-2 --wd2 1e-2

# Pubmed
python main.py --dataname pubmed --epochs 100 --lambd 1e-3 --dfr 0.3 --der 0.5 --lr2 1e-2 --wd2 1e-4

# Amazon-Computer
python main.py --dataname comp --epochs 50 --lambd 5e-4 --dfr 0.1 --der 0.3 --lr2 1e-2 --wd2 1e-4

# Amazon-Photo
python main.py --dataname photo --epochs 50 --lambd 1e-3 --dfr 0.2 --der 0.3 --lr2 1e-2 --wd2 1e-4

# Coauthor-CS
python main.py --dataname cs --epochs 50 --lambd 1e-3 --dfr 0.2 --lr2 5e-3 --wd2 1e-4 --use_mlp

# Coauthor-Physics
python main.py --dataname physics --epochs 100 --lambd 1e-3 --dfr 0.5 --der 0.5 --lr2 5e-3 --wd2 1e-4