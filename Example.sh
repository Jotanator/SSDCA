# Train SSDCA (Siamese Swin with Dual Cross-Attention)
# For 30 Epochs, Using Fold 2, Balanced sampling

python main.py --num_classes 2 \
--model_config SSDCA \
--num_epochs 30 \
--batch_size 8 \
--learning_rate 2e-5 \
--data_dir /path/to/data/directory/ \
--json_fold /path/to/fold1.json \
--sampler balanced \
--save_dir "checkpoints" \
--model_naming SSDCA_Fold1 \
--seed 115
