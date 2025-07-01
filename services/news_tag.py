import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("services/news_tag")
model = AutoModelForSequenceClassification.from_pretrained("services/news_tag")
model.eval()


def classify_news(text: str) -> int:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=128)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0]

        max_prob = probs.max().item()
        pred_label = probs.argmax().item()  # 0-based

        if max_prob < 0.4:
            return 0  # 확률 낮아서 기타
        else:
            return pred_label + 1  # 1-based 클래스 (1, 2, 3)
