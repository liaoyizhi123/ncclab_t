# CineBrain 要修改格式

from pathlib import Path
import mne
import numpy as np
import pandas as pd
import h5py

root_path = Path(__file__).resolve().parent
eeg_base_dir = Path('/mnt/dataset2/Datasets/CineBrain_EEG_Raw')
target_sfreq = 200.0
window_size = 2.0
stride_size = 2.0  # FIXME. stride_size没用上
label_path = Path('/home/liaoyizhi/codes/ncclab_t/results/smooth_scores')

h5_dir_root = Path(f'/mnt/dataset2/Processed_datasets/EEG_Bench')
h5_dir_path = h5_dir_root / f"CineBrain_hdf5_T={window_size}s_stride={stride_size}"
h5_dir_path.mkdir(parents=True, exist_ok=True)

label_mapping = {
    '0701': '000000_000001-000268_000269',
    '0702': '000270_000271-000538_000539',
    '0703': '000540_000541-000808_000809',
    '0704': '000810_000811-001078_001079',
    '0705': '001080_001081-001348_001349',
    '0706': '001350_001351-001618_001619',
    '0707': '001620_001621-001888_001889',
    '0708': '001890_001891-002158_002159',
    '0709': '002160_002161-002428_002429',
    '0710': '002430_002431-002698_002699',
    '0901': '002700_002701-002968_002969',
    '0902': '002970_002971-003238_003239',
    '0903': '003240_003241-003508_003509',
    '0904': '003510_003511-003778_003779',
    '0905': '003780_003781-004048_004049',
    '0906': '004050_004051-004318_004319',
    '0907': '004320_004321-004588_004589',
    '0908': '004590_004591-004858_004859',
    '0909': '004860_004861-005128_005129',
    '0910': '005130_005131-005398_005399',
    '1101': '005400_005401-005668_005669',
    '1102': '005670_005671-005938_005939',
    '1103': '005940_005941-006208_006209',
    '1104': '006210_006211-006478_006479',
    '1105': '006480_006481-006748_006749',
    '1106': '006750_006751-007018_007019',
    '1107': '007020_007021-007288_007289',
    '1108': '007290_007291-007558_007559',
    '1109': '007560_007561-007828_007829',
    '1110': '007830_007831-008098_008099',
}


def show(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[G] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"[D] {name}  shape={obj.shape} dtype={obj.dtype}")
        if len(obj.attrs) > 0:
            print("    attrs:", list(obj.attrs.keys()))


sub_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06']
for sub_idx, sub in enumerate(sub_list):
    with h5py.File(str(h5_dir_path / f'sub_{sub_idx}.h5'), 'w') as h5f_root:
        subject_attrs_set = False

        # iterate episodes/trials/videos
        for video_idx, video_path in enumerate(sorted(list((eeg_base_dir / sub).glob("*.mff")))):
            # create trial group
            trial_group = h5f_root.create_group(f'trial{video_idx}')
            trial_group.attrs['trial_id'] = video_idx
            trial_group.attrs['session_id'] = 'None'  # FIXME.
            video_name = video_path.stem.split('_')[-1]

            raw = mne.io.read_raw_egi(str(video_path), preload=True)  # 读取 EGI .mff
            montage = raw.get_montage()
            if montage is not None:
                ch_pos_dict = montage.get_positions().get('ch_pos', {})
            else:
                ch_pos_dict = {}

            _old_sfreq = raw.info['sfreq']
            events = mne.find_events(raw, stim_channel='TREV', verbose=False)
            if _old_sfreq != target_sfreq:
                raw.resample(target_sfreq, verbose=False)
                events[:, 0] = np.round(events[:, 0] * target_sfreq / _old_sfreq).astype(int)

            assert len(events) >= 1350
            events = events[:1350]
            markers_arr = events[:, 0]

            # picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            # picks64 = picks
            # channel_li = [raw.ch_names[i] for i in picks64]

            raw.drop_channels(['VREF', 'TREV', 'ECG'])  # 去掉眼电等伪迹通道
            channel_li = raw.ch_names
            sfreq = raw.info['sfreq']
            if ch_pos_dict:
                chn_pos = np.array(
                    [ch_pos_dict.get(name, [np.nan, np.nan, np.nan]) for name in channel_li],
                    dtype=float,
                )
            else:
                chn_pos = None
            if not subject_attrs_set:
                h5f_root.attrs['subject_id'] = sub_idx
                h5f_root.attrs['dataset_name'] = 'CineBrain'
                h5f_root.attrs['task_type'] = 'None'  # FIXME.
                h5f_root.attrs['downstream_task_type'] = 'None'  # FIXME.
                h5f_root.attrs['rsFreq'] = float(sfreq)
                h5f_root.attrs['chn_name'] = channel_li
                if chn_pos is None:
                    h5f_root.attrs['chn_pos'] = 'None'
                else:
                    h5f_root.attrs['chn_pos'] = chn_pos
                h5f_root.attrs['chn_ori'] = 'None'
                h5f_root.attrs['chn_type'] = 'EEG'
                h5f_root.attrs['montage'] = '10_20'
                subject_attrs_set = True

            # if int(markers_li[-1] + 800 - markers_li[0]) / 1000.0 != 18*60.0:
            #     continue
            assert (
                int(markers_arr[-1] + (800 * target_sfreq / _old_sfreq) - markers_arr[0]) / sfreq == 18 * 60.0
            ), f"{sub} {video_path.name} 事件数不对"

            start = int(markers_arr[0])
            step = int(round(window_size * sfreq))
            stop = start + int(round(18 * 60.0 * sfreq))

            new_markers = np.arange(start, stop, step)

            new_events = np.zeros((new_markers.shape[0], 3), dtype=int)
            new_events[:, 0] = new_markers
            new_events[:, 2] = 1

            # ## slice
            epochs = mne.Epochs(raw, new_events, tmin=0, tmax=window_size, baseline=None, preload=True)
            data = epochs.get_data()[:, :, :-1]  # (n_epochs, n_channels, n_times)

            # ###################### get labels
            assert video_name in label_mapping.keys()
            label_file_name = label_mapping[video_name]
            label_file_path = list(label_path.glob(f'*{label_file_name}*'))
            assert len(label_file_path) == 1

            # 使用pandas read csv
            labels_df = pd.read_csv(label_file_path[0], header=0)
            labels_np = labels_df.to_numpy()[:, 1:]  # (n_times, n_label_dims)
            # ###################### get labels

            assert data.shape[0] == labels_np.shape[0]
            # iterate segments
            for segment_idx, (data_segment, label_segment) in enumerate(zip(data, labels_np)):
                # create sample group
                segment_group = trial_group.create_group(f'segment{segment_idx}')
                dataset = segment_group.create_dataset('eeg', data=data_segment, compression="gzip")

                start_time = segment_idx * window_size
                end_time = start_time + window_size
                dataset.attrs['segment_id'] = segment_idx
                dataset.attrs['start_time'] = float(start_time)
                dataset.attrs['end_time'] = float(end_time)
                dataset.attrs['time_length'] = float(window_size)
                dataset.attrs['label'] = label_segment

            # pass  # segments end
            # h5f_root.visititems(show)
            # data = h5f_root['trial0/sample0/eeg'][:]
            pass  # segments end

        pass  # subject end
    pass  # subject end
pass  # all subj end
