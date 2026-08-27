import albumentations as A


def get_pipeline_geometric(prob=0.5, size=224):
    pipeline = A.Compose([
        A.VerticalFlip(p=prob),
        A.HorizontalFlip(p=prob),
        A.RandomRotate90(p=prob),
        A.RandomResizedCrop(size=(size, size), scale=(0.80, 0.95), interpolation=2, p=prob),
    ])
    return pipeline


def get_pipeline_color(prob=0.5):
    pipeline = A.Compose([
        A.RGBShift(r_shift_limit=(-50, 10), g_shift_limit=(-50, 10), b_shift_limit=(-50, 10), p=prob),
    ])
    return pipeline
