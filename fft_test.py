from PIL import Image, ImageDraw


def create_bw_bmp(
    output_filename: str = "output.bmp",
    image_size: tuple[int, int] = (128, 128),
    square_size: tuple[int, int] = (4, 4),
    square_position: tuple[int, int] | None = None,
):
    """
    构造一幅黑白二值BMP图像。

    在指定大小的黑色背景上，于指定位置产生一个指定大小的白色方块。

    Args:
        output_filename (str): 输出的BMP文件名。
        image_size (tuple[int, int]): 图像的 (宽度, 高度)。
        square_size (tuple[int, int]): 白色方块的 (宽度, 高度)。
        square_position (tuple[int, int] | None): 白色方块左上角的 (x, y) 坐标。如果为 None，则将方块置于图像中心。
    """
    img_width, img_height = image_size
    sq_width, sq_height = square_size

    img = Image.new("1", (img_width, img_height), 0)

    if square_position is None:
        top_left_x = (img_width - sq_width) // 2
        top_left_y = (img_height - sq_height) // 2
    else:
        top_left_x, top_left_y = square_position

    bottom_right_x = top_left_x + sq_width
    bottom_right_y = top_left_y + sq_height

    if (
        top_left_x < 0
        or top_left_y < 0
        or bottom_right_x > img_width
        or bottom_right_y > img_height
    ):
        print(
            f"警告: 方块位置 {((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))} 可能超出图像边界 {image_size}。"
        )

    draw = ImageDraw.Draw(img)

    box = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

    draw.rectangle(box, fill=1)

    try:
        img.save(output_filename, "BMP")
        print(f"成功创建图像: '{output_filename}'")
        print(f" - 图像尺寸: {image_size}")
        print(f" - 方块尺寸: {square_size}")
        print(f" - 方块左上角位置: ({top_left_x}, {top_left_y})")
    except Exception as e:
        print(f"保存文件时出错: {e}")


if __name__ == "__main__":
    create_bw_bmp("img/center.bmp")
    create_bw_bmp(output_filename="img/translation.bmp", square_position=(16, 16))
    create_bw_bmp(output_filename="img/small.bmp", square_size=(1, 1))
    create_bw_bmp(output_filename="img/big.bmp", square_size=(32, 32))
    create_bw_bmp(output_filename="img/large.bmp", square_size=(64, 64))
