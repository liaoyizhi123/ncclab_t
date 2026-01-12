import h5py

if __name__ == "__main__":
    file_path = f'/home/liaoyizhi/codes/ncclab_t/CineBrain_hdf5_T=2.0s_stride=2.0.h5'

    def print_structure(name, obj):
        """递归打印 HDF5 节点的名称和类型"""
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name} (Shape: {obj.shape}, Type: {obj.dtype})")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}Group: {name}")

    def walk_and_count(name, obj):
        level = name.count('/')
        indent = "  " * level

        if isinstance(obj, h5py.Group):
            # 获取该组下直接子项的数量
            num_children = len(obj.keys())
            print(f"{indent}Group: {name} (including {num_children} direct children)")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}Dataset: {name} (Shape: {obj.shape}, Type: {obj.dtype})")

    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)
        # f.visititems(walk_and_count)
