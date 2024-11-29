# LLM Project

## Project Task
I chose sentiment analysis using the IMDB database.

## Dataset
I used the IMDB dataset that contains reviews and then labels (0 ,1) for positive or negative reviews.

## Pre-trained Model
I used a model called reviews-sentiment-analysis 
(https://huggingface.co/juliensimon/reviews-sentiment-analysis)

Distilbert model fine-tuned on English language product reviews

A notebook for Amazon SageMaker is available in the 'code' subfolder.

## Performance Metrics
Loss: 0.2832
Accuracy: 0.9321
F1: 0.9320

I chose accuracy and F1 scores because our readings said these were the best for sentiment analysis. 

## Hyperparameters
   output_dir="my_tuned_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps = 500,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    metric_for_best_model="accuracy",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
    report_to="none"

I ended up buying google credits to run this on their A100 GPU because the num_epochs made a huge difference to the model. 

Here is a link to my Model Report 
(https://huggingface.co/landoncodes/my_awesome_model)