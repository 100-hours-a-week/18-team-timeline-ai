def next_timeline_type(current: str) -> str:
    order = ["DAY", "WEEK", "MONTH"]
    if current not in order:
        return "ERROR"

    index = order.index(current)
    return order[min(index + 1, len(order) - 1)]
