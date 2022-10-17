def calculate_autosize(max_size: int, *sizes: int) -> tuple[list[int], int]:
    sizes = list(sizes)
    sum_size = sum(sizes)
    for i in range(len(sizes)):
        if sizes[i] == -1:
            sizes[i] = max_size - sum_size - 1
            sum_size = max_size
            break
    if sum_size > max_size:
        raise ValueError("The sum of 'sizes' exceeds 'max_size'")
    if any([size < 0 for size in sizes]):
        raise ValueError(
            "Only one of the 'sizes' could be -1 and all others must be positive or zero")
    return sizes, sum_size


def format_time(seconds) -> str:
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    if days > 0:
        output = f"{days}D{hours:0>2d}h{minutes:0>2d}m"
    elif hours > 0:
        output = f"{hours:0>2d}h{minutes:0>2d}m{seconds_final:0>2d}s"
    elif minutes > 0:
        output = f"{minutes:0>2d}m{seconds_final:0>2d}s"
    elif seconds_final > 0:
        output = f"{seconds_final:0>2d}s{millis:0>3d}ms"
    elif millis > 0:
        output = f"{millis:0>3d}ms"
    else:
        output = '0ms'
    return output
