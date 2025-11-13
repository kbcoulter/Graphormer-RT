### NOTE: THIS SCRIPT IS NOT USED IN deep_metab APPTAINER USE (NOT RUN FROM app_evaluate_RP.sh) ###
### TO EDIT OPTIONS, PLEASE CD: deep_metab/app_evaluate_RP.sh ###
echo "NOTE: THIS SCRIPT IS FUNCTIONING OUTSIDE OF deep_metab/app_evaluate_RP.sh AND EXPECTED HPC WORKFLOW. CONTINUING..." 

cd ./Graphormer-RT/graphormer/evaluate/

python evaluate.py \
    --user-dir ../../graphormer \
    --num-workers 32 \
    --ddp-backend=legacy_ddp \
	--user-data-dir rp_test \
	--dataset-name RT_test \
    --task graph_prediction \
	--criterion rmse \
	--arch graphormer_base \
    --encoder-layers 8 \
    --encoder-embed-dim  512 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 64 \
    --freeze-level -4 \
    --mlp-layers 5 \
    --batch-size 64 \
    --num-classes 1 \
    --save-path '../../../predictions/RP_preds.csv' \
    --save-dir '/workspace/Graphormer-RT/checkpoints/' \
    --split train \