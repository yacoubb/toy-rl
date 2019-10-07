import os
import cv2

step = 5


def create_gif(input_path):
    all_paths = os.listdir(input_path)
    # grab all image paths in the input directory
    image_paths = []
    for img_path in all_paths:
        if img_path.endswith(".jpg"):
            index = int(img_path.split("/")[-1][:-4])
            if index % step == 0:
                image_paths.append(input_path + img_path)

    image_paths = sorted(image_paths, key=lambda x: int(x.split("/")[-1][:-4]))
    print(f"got {len(image_paths)} images")

    temp_path = f'./__temp/{input_path.split("/")[-2]}'
    print(temp_path)
    from shutil import rmtree

    if os.path.isdir(temp_path):
        rmtree(temp_path)
    os.mkdir(temp_path)

    converted_paths = []
    for i in range(len(image_paths)):
        label = image_paths[i].split("/")[-2]
        gen = label.split("_")[1]
        frame = cv2.imread(image_paths[i])
        width = frame.shape[1]
        frame = cv2.putText(frame, f"generation: {gen}", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
        frame = cv2.putText(frame, f"step: {i}", (width - 250 - len(str(i)) * 25, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
        cv2.imwrite(f"{temp_path}/{i}.png", frame)
        converted_paths.append(f"{temp_path}/{i}.png")

    output_path = f"./{len(converted_paths)}.gif"

    joined_paths = " ".join(converted_paths)
    cmd = f"convert -delay 0 {joined_paths} {output_path}"
    os.system(cmd)

    rmtree(temp_path)


if not os.path.isdir("./__temp"):
    os.mkdir("./__temp")

from multiprocessing import Pool, cpu_count

print(f"found {cpu_count()} cpus, starting {cpu_count() - 2} workers")
with Pool(cpu_count() - 2) as p:
    folders = os.listdir("./renders")
    folders = list(map(lambda x: f"./renders/{x}/", folders))
    folders = list(filter(lambda x: os.path.isdir(x), folders))
    print(folders)
    p.map(create_gif, folders)
