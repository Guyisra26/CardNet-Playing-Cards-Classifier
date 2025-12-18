
from fastai.vision.all import *
from pathlib import Path
from .config import DataConfig

class CardDataModule:
    def __init__(self, config: DataConfig):
        self.cfg = config
        self.dls = None

    def setup(self):
        data_path = self.cfg.base_path
        batch_tfms = aug_transforms(
            mult=1.0,
            do_flip=True,
            flip_vert=False,
            max_rotate=15,
            max_zoom=1.1,
            max_lighting=0.2,
            max_warp=0.1,
        )

        self.dls = ImageDataLoaders.from_folder(
            data_path,
            train=self.cfg.train_dir,
            valid=self.cfg.valid_dir,
            valid_pct = None,
            bs = self.cfg.batch_size,
            seed = self.cfg.seed,
            item_tfms=Resize(self.cfg.img_size, method='squish'),
            batch_tfms=batch_tfms
        )

    def verify(self, sample_size: int = 800, remove: bool = False):
        import random
        print("Verifying dataset integrity...")

        # ✅ בדוק רק את תיקיית ה-train (במקום כל ה-base_path)
        path_to_check = self.cfg.base_path / self.cfg.train_dir

        files = get_image_files(path_to_check)

        # ✅ דגימה כדי שזה לא יקח שעה
        if sample_size is not None and len(files) > sample_size:
            files = random.sample(list(files), sample_size)

        failed = verify_images(files)

        print(f"Found {len(failed)} corrupted images (checked {len(files)} files).")

        # ✅ למחוק רק אם מבקשים
        if remove:
            for f in failed:
                print(f"Deleting bad file: {f}")
                Path(f).unlink(missing_ok=True)  # py>=3.8

        return failed

    def get_dls(self):
        if self.dls is None:
            self.setup()
        return self.dls

