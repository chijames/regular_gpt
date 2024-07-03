python training/example.py --task aaaa --seed 42 --steps 100000 --lr 1e-5 --embedding_dim 256 --num_heads 8
python training/example.py --task aa --seed 42 --steps 10000 --lr 5e-4 --embedding_dim 256 --num_heads 8
python training/example.py --task abab --seed 42 --steps 10000 --lr 1e-5 --embedding_dim 256 --num_heads 8
python training/example.py --task modular_arithmetic --seed 42 --steps 500000 --lr 3e-4 --embedding_dim 1024 --num_heads 8 --mod 3
python training/example.py --task modular_arithmetic --seed 40 --steps 1000000 --lr 1e-4 --embedding_dim 256 --mod 5
python training/example.py --task cycle_navigation --seed 42 --steps 10000 --lr 5e-4 --embedding_dim 256 --num_heads 8
python training/example.py --task d_12 --seed 42 --steps 50000 --lr 1e-4 --sequence_length 80
python training/example.py --task d_1 --seed 42 --steps 100000 --lr 1e-4
python training/example.py --task d_2 --seed 42 --steps 100000 --lr 5e-4
python training/example.py --task d_3 --seed 42 --steps 100000 --lr 1e-4
python training/example.py --task d_4 --seed 42 --steps 100000 --lr 1e-4
python training/example.py --task even_pairs --seed 42 --steps 10000 --lr 3e-5 --embedding_dim 256 --num_heads 8
python training/example.py --task parity_check --seed 42 --steps 100000 --lr 3e-4
python training/example.py --task tomita_1 --seed 42 --steps 100000 --lr 5e-4 --max_range_test_length 100
python training/example.py --task tomita_3 --seed 42 --steps 100000 --lr 5e-4 --max_range_test_length 100
python training/example.py --task tomita_4 --seed 42 --steps 100000 --lr 5e-4 --max_range_test_length 100
python training/example.py --task tomita_5 --seed 42 --steps 100000 --lr 1e-4 --max_range_test_length 100
python training/example.py --task tomita_6 --seed 42 --steps 100000 --lr 5e-4 --max_range_test_length 100
python training/example.py --task tomita_7 --seed 42 --steps 100000 --lr 5e-4 --max_range_test_length 100
