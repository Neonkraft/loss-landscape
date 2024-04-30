SCRIPT_DIR="vscode_remote_debugging"
while read var value
    do
        export "$var"="$value"
    done < $SCRIPT_DIR/config.conf


# mpirun -n 1 python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client plot_surface.py --mpi --cuda --model resnet56 --x=-1:1:51 --y=-1:1:51 \
# --model_file cifar10/trained_nets/resnet56_sgd_lr=0.1_bs=128_wd=0.0005/model_300.t7 \
# --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot

mpirun -n 1 python -m debugpy --listen 0.0.0.0:$PORT --wait-for-client plot_surface.py --mpi --cuda --model resnet56 --x=-1:1:51 --y=-1:1:51 \
--model "nb201" --model_file 2024-04-29-12:42:04.012/checkpoints/model_0000003.pth \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot




