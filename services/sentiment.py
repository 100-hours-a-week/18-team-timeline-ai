import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("services/sentiment")
model = AutoModelForSequenceClassification.from_pretrained("services/sentiment")
model.eval()


def classify_comment(text: str) -> str:
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=128)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)[0][1].item()  # 긍정일 확률

        if probs < 0.1:
            return "부정"
        elif probs > 0.9:
            return "긍정"
        else:
            return "중립"


async def classify_comment_async(text: str) -> str:
    return classify_comment(text)
