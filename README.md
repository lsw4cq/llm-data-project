# LLM Project

## Project Task
I chose sentiment analysis using the IMDB database.

## Dataset
I used the IMDB dataset that contains reviews and then labels (0 ,1) for positive or negative reviews.

## Pre-trained Model
(fill in details about the pre-trained model you selected)

## Performance Metrics
Loss: 0.2832
Accuracy: 0.9321
F1: 0.9320

I chose accuracy and F1 scores because our readings said these were the best for sentiment analysis. 

## Hyperparameters

learning_rate: 3e-05
train_batch_size: 16
eval_batch_size: 16
seed: 42
optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
lr_scheduler_type: linear
lr_scheduler_warmup_steps: 500
num_epochs: 3
mixed_precision_training: Native AMP

I ended up buying google credits to run this on their A100 GPU because the num_epochs made a huge difference to the model. 