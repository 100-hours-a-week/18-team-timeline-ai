def download_model(repo: str = "username/model_name", save_dir: str = "services/dir_name"):
    import os
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    os.makedirs(save_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)


download_model("M0N0RA1L/news_tag", "services/news_tag")
download_model("M0N0RA1L/sentiment-model", "services/sentiment")

# In the root directory, do this:
# PYTHONPATH=. python scripts/download_open_model.py
