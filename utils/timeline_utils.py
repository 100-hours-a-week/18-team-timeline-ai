def next_timeline_type(current: str) -> str:
    order = ["DAY", "WEEK", "MONTH"]
    if current not in order:
        return "ERROR"

    index = order.index(current)
    return order[min(index + 1, len(order) - 1)]


def convert_tag(tag: str) -> int:
    if "경제" in tag:
        return 1
    elif "연예" in tag:
        return 2
    elif "스포츠" in tag:
        return 3
    else:
        return 0


def extract_first_sentence(text: str) -> str:
    if "." in text:
        return text.split(".")[0].strip() + "."
    return text.strip()
