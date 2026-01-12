import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

DEFAULT_CITY = "NY"

class RiskMap(Dataset):
    def __init__(self, root="./data", **kwargs):
        """
        root: 数据根目录 (包含 NY/LA 子文件夹)
        """
        super().__init__()
        self.root = root

        # 固定使用 DEFAULT_CITY
        self.city = DEFAULT_CITY
        ds_base = os.path.join(self.root, self.city, "DataSets")
        if not os.path.exists(ds_base):
            raise FileNotFoundError(f"DataSets 目录不存在: {ds_base} （请检查 DEFAULT_CITY 是否设置正确并确保目录存在）")

        # 读取文件名列表（由子类实现）
        self.filenames = self.getFileNames()
        if not isinstance(self.filenames, (list, tuple)):
            raise ValueError("getFileNames() must return a list of file base names (no extension).")
        self.labels = {"file_names": self.filenames}
        self._length = len(self.filenames)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        base = self.labels["file_names"][i]
        example = {"file_name": base}

        ds_base = os.path.join(self.root, self.city, "DataSets")
        RISKMAP_PATH_His = os.path.join(ds_base, "192021")
        RISKMAP_PATH_Obs = os.path.join(ds_base, "222324")
        CON_IMG_MAP_512 = os.path.join(ds_base, "MapTiles")
        CON_IMG_SATELLITE_512 = os.path.join(ds_base, "SatelliteTiles")
        POI_FOLDER = os.path.join(ds_base, "POI")

        fn = base + ".npy"

        path_his = os.path.join(RISKMAP_PATH_His, fn)
        if not os.path.exists(path_his):
            raise FileNotFoundError(f"riskmap (192021) not found: {path_his}")
        # example["riskmap_his"] = np.load(path_his)[None].astype(np.float32)
        example["riskmap_his"] = np.load(path_his).astype(np.float32)

        path_obs = os.path.join(RISKMAP_PATH_Obs, fn)
        if not os.path.exists(path_obs):
            raise FileNotFoundError(f"riskmap (222324) not found: {path_obs}")
        # example["riskmap_obs"] = np.load(path_obs)[None].astype(np.float32)
        example["riskmap_obs"] = np.load(path_obs).astype(np.float32)

        map_path = os.path.join(CON_IMG_MAP_512, base + ".png")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Map tile not found: {map_path}")
        example["map"] = self.preprocess_image(map_path)

        sat_path = os.path.join(CON_IMG_SATELLITE_512, base + ".png")
        if not os.path.exists(sat_path):
            raise FileNotFoundError(f"Satellite tile not found: {sat_path}")
        example["satellite"] = self.preprocess_image(sat_path)

        poi_path = os.path.join(POI_FOLDER, fn)
        if not os.path.exists(poi_path):
            raise FileNotFoundError(f"POI file not found: {poi_path}")
        example["poi"] = np.load(poi_path).astype(np.float32)

        return example

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        arr = np.array(image).astype(np.float32)
        return (arr / 127.5 - 1.0)

    def getFileNames(self):
        return []

class RiskMapTrain(RiskMap):
    def __init__(self, root="./data", **kwargs):
        self.split_name = "train.txt"
        super().__init__(root=root, **kwargs)

    def getFileNames(self):
        path = os.path.join(self.root, self.city, "DataSets", self.split_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Train list not found: {path}")
        with open(path, "r") as f:
            return [ln.strip() for ln in f.readlines() if ln.strip()]

class RiskMapValidation(RiskMap):
    def __init__(self, root="./data", **kwargs):
        self.split_name = "val.txt"
        super().__init__(root=root, **kwargs)

    def getFileNames(self):
        path = os.path.join(self.root, self.city, "DataSets", self.split_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Val list not found: {path}")
        with open(path, "r") as f:
            return [ln.strip() for ln in f.readlines() if ln.strip()]

class RiskMapTest(RiskMap):
    def __init__(self, root="./data", **kwargs):
        self.split_name = "test.txt"
        super().__init__(root=root, **kwargs)

    def getFileNames(self):
        path = os.path.join(self.root, self.city, "DataSets", self.split_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Test list not found: {path}")
        with open(path, "r") as f:
            return [ln.strip() for ln in f.readlines() if ln.strip()]
