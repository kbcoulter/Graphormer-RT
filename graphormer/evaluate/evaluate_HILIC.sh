### NOTE: THIS SCRIPT IS NOT USED IN deep_metab APPTAINER USE (NOT RUN FROM app_evaluate_HILIC.sh) ###
### TO EDIT OPTIONS, PLEASE CD: deep_metab/app_evaluate_HILIC.sh ###

echo "This script is functioning outside of the expected deep_metab workflow."
echo "Please: cd ../../.. and evaluate via app_evaluate_HILIC.sh"
echo "Exiting..."
exit 

python evaluate_HILIC.py \
    --user-dir ../../graphormer \
    --num-workers 32 \
    --ddp-backend=legacy_ddp \
	--user-data-dir hilic_test \
	--dataset-name HILIC_a \
    --task graph_prediction \
	--criterion rmse \
	--arch graphormer_HILIC \
    --encoder-layers 8 \
    --encoder-embed-dim  512 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 64 \
    --mlp-layers 5 \
    --batch-size 64 \
    --num-classes 1 \
    --save-path 'None' \
    --save-dir '../../checkpoints' \
    --split train \