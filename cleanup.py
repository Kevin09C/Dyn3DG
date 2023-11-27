import os


for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
    for i in range(0, 31):
        path = f'data/{sequence}/epipolar_error_png/{i}'
        files = sorted(os.listdir(path))
        for index, file in enumerate(files):
            os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index).zfill(6), '.png'])))
