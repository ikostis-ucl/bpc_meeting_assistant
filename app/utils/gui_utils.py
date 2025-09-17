def generate_colors(count):
    if count == 0:
        return []
    elif count == 1:
        return ["#1976D2"]
    shades = []
    for i in range(count):
        intensity = 0.3 + (0.6 * i / (count - 1))
        r = int(25 + (135 * intensity))
        g = int(118 + (82 * intensity))
        b = int(210 + (45 * intensity))
        shades.append(f"#{r:02x}{g:02x}{b:02x}")
    return shades


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
