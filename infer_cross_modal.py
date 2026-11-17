"""BioCross 单样本推理：II_Pleth_ABP / hwd，并保存重建波形图。

模型使用范围（当前默认权重）：
    - 权重由MIMIC-IV-waveform数据集训练得到。
    - 本脚本支持 PPG → ECG / ABP / 两者，以及 ECG + PPG → ABP。
    - data/II_Pleth_ABP 为 MIMIC-IV-waveform 的测试集，包含同步的 ECG、PPG、ABP 波形及 metadata。
    - data/hwd 为相机 rPPG 的跨来源输入适配，可生成 ECG、ABP；该数据没有同步真值。
      输出心率、血压是从模型生成波形提取的估计值，适配不等于已验证的泛化性能。

输入输出的时长与采样规格：
    - 当前权重按每段 10 秒、每通道 512 点训练，模型端等效采样率为 51.2 Hz。
      原训练 merged_data 每通道 624 点，经 Dt.py 重采样为 512 点后进入模型。
    - II_Pleth_ABP：Test_data.npy 应为 (N, 3, 512)，通道顺序为 ECG、PPG、ABP。
      本脚本直接读取已处理的片段，不会再次重采样此文件；对应 metadata 为 (N, 4)。
    - hwd：wave_30hz.csv 的 rPPG 为 30 Hz；默认截取 10 秒（约 300 点），
      再重采样为 512 点输入模型。--start 相对于 CSV 第一条波形的时间。
    - 模型输入张量为 (1, 1, 512)（PPG）或 (1, 2, 512)（ECG、PPG）。
      每个重建模态单独保存为 (1, 1, 512) 的 .npy，默认同为 10 秒、51.2 Hz。
    - 部署现有权重时使用 --duration 10。修改 hwd 的 duration 仍会重采样到 512 点，
      此时等效采样率为 512 / duration Hz，时间尺度已偏离原训练设置，需要另行验证。
      对 II_Pleth_ABP，duration 仅影响时间轴及 HR/BP 提取所用的采样率，不会截取片段。
      更改配置中的 num_samples 需要匹配相应权重，不能直接改变当前权重的输出长度。

幅值与 metadata：
    - 训练 ECG、PPG 保留源数据幅值，没有逐段 Min-Max / Z-score；ECG 源单位为 mV。
      训练 ABP 除以 100；本脚本将重建 ABP 乘回 100，以 mmHg 保存并提取血压。
      hwd 的 rPPG 另做逐片段 Min-Max 到 [0, 1]，这是新增的输入适配步骤。
    - metadata 顺序为 [时间 sin, 时间 cos, 归一化年龄, 性别]，时间取本地整数小时。
      原训练年龄范围为 26–91 岁；本脚本将越界年龄限制在边界，男 1 / 女 0。

参数说明（命令在项目根目录执行，相对路径以当前工作目录为基准）：
    --task / --dataset {II_Pleth_ABP,hwd}
        数据集名称，默认读取配置文件中的 task。
    --config PATH
        模型结构及重采样配置，默认 config/TriplemetaVAE3.yaml，需与权重匹配。
    --data-dir DIR
        数据根目录，默认 data；脚本读取 DIR/<task>/ 下的数据。
    --checkpoint PATH
        完整 BioCross 模型权重。默认 checkpoints/<task>/model.pth；
        hwd 则使用 checkpoints/<配置中的 task>/model.pth。
    --sample-index INT
        从 0 开始的样本编号，默认 0。II_Pleth_ABP 对应 Test_data.npy 的行；
        hwd 对应按目录名排序的有效记录，提供 --record 时优先使用指定记录。
    --record NAME
        仅 hwd 使用，指定记录子目录名，如 20261116_211644；默认不指定。
    --start SECONDS
        仅 hwd 使用，窗口起点相对于 CSV 第一条波形，默认 0 秒。
    --duration SECONDS
        窗口时长，默认 10 秒，必须大于 0。hwd 按此时长截取；
        II_Pleth_ABP 不重新截取，仅据此设置时间轴及 HR/BP 提取采样率。
    --ppg-column {raw,filtered}
        仅 hwd 使用，读取 CSV 的哪一列作为 rPPG，默认 filtered。
    --known-modality {PPG,ECG_PPG}
        模型输入，默认 PPG；ECG_PPG 表示依次输入 ECG、PPG。hwd 仅支持 PPG。
    --target-modality {ECG,ABP,both}
        重建目标；both 同时输出 ECG、ABP。未指定时，PPG 输入默认输出 ECG，
        ECG_PPG 输入默认输出 ABP。输出含 ECG 时提取心率，含 ABP 时提取血压。
    --age YEARS / --gender {0,1}
        hwd 的实际年龄（岁）和性别（女 0 / 男 1），默认不指定；
        未提供 --meta 时，hwd 必须同时提供二者。II_Pleth_ABP 自动读取 metadata。
    --age-range MIN MAX
        年龄归一化及截断范围，默认 26 91，满足 0 <= MIN < MAX；应匹配训练预处理。
    --meta TIME_SIN TIME_COS AGE_NORM SEX
        可选，直接覆盖 4 个已编码的 metadata 值，优先于自动读取/生成；
        AGE_NORM 仍会限制在 [0, 1]。原始年龄需先按训练范围编码后传入此参数。
    --output-dir DIR
        输出目录，默认 output/<task>/single_sample/。
    --show
        可选开关，默认关闭；加上后保存图片并弹窗显示，关闭窗口后继续。
    -h / --help
        显示本说明和命令行参数帮助后退出。

使用示例：
    # 在项目根目录执行；sample-index 从 0 开始。
    # II_Pleth_ABP：输入 PPG，同时重建 ECG 和 ABP，自动读取 metadata。
    python infer_cross_modal.py --task II_Pleth_ABP --sample-index 0 --target-modality both

    # 仅重建 ECG；仅重建 ABP 时将 ECG 改为 ABP。
    python infer_cross_modal.py --task II_Pleth_ABP --sample-index 10 --target-modality ECG

    # 输入 ECG + PPG，重建 ABP。
    python infer_cross_modal.py --task II_Pleth_ABP --sample-index 0 --known-modality ECG_PPG --target-modality ABP

    # hwd：时间自动读取 summary.json，手动输入年龄和性别（男 1 / 女 0）。
    # 年龄自动限制在训练范围 26–91 岁，图中同时标注输入年龄和模型使用年龄。
    python infer_cross_modal.py --task hwd --record 20261116_211644 --target-modality both --age 22 --gender 1

    # hwd：从波形起点后第 5 秒开始截取 10 秒，使用 raw 列。
    python infer_cross_modal.py --task hwd --record 20261116_211644 --start 5 --duration 10 --ppg-column raw --target-modality both --age 35 --gender 0

    # 自定义权重和输出目录，并弹窗显示图片。
    python infer_cross_modal.py --task II_Pleth_ABP --sample-index 0 --target-modality both --checkpoint checkpoints/II_Pleth_ABP/model.pth --output-dir output/demo --show

默认权重：checkpoints/II_Pleth_ABP/model.pth（hwd 复用此权重）。
默认输出：output/<task>/single_sample/，包含重建 .npy、波形 .png 和合并统计 *_stats.json。
统计中 hr 为心率及差值，bp 为 SBP/DBP 及误差；hwd 没有真实 ECG/ABP，不能计算相应真值误差。
"""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
import torch
import yaml

from model.BioCross import TripleVAE_cmeta3_fft
import utils.metrics as metrics
from utils.phenotype import caculate_abp_phenotype, calculate_heart_rate

TARGET_MODALITIES = {'ECG': ('ECG',), 'ABP': ('ABP',), 'both': ('ECG', 'ABP')}


def load_single_sample(args, num_samples):
    """返回 (1, C, L) 输入、(1, 4) 元数据、可选的真实目标波形。"""
    data_dir = Path(args.data_dir) / args.task
    age_min, age_max = args.age_range
    if not np.isfinite([age_min, age_max]).all() or not 0 <= age_min < age_max:
        raise ValueError('年龄归一化范围须为有限数，且满足 0 <= MIN < MAX。')
    args.timestamp = None
    if args.task == 'hwd':
        if args.known_modality != 'PPG':
            raise ValueError('hwd 只有 PPG，known_modality 请使用 PPG。')
        records = sorted(data_dir.glob('*/wave_30hz.csv'))
        if args.record:
            path = data_dir / args.record / 'wave_30hz.csv'
        else:
            if not 0 <= args.sample_index < len(records):
                raise IndexError(f'hwd 共 {len(records)} 条记录，sample-index 超出范围。')
            path = records[args.sample_index]
        wave = np.genfromtxt(path, delimiter=',', names=True)
        time = wave['t_s'] - wave['t_s'][0]
        dt = np.median(np.diff(time))
        end = args.start + args.duration
        if args.start < 0 or time[-1] + dt < end - 1e-3:
            raise ValueError('所选记录不足以截取指定的 start / duration。')
        ppg = wave[args.ppg_column][(time >= args.start) & (time < end)]
        if len(ppg) < 2 or not np.isfinite(ppg).all():
            raise ValueError('所选 PPG 片段过短或包含非有限值。')
        ppg = resample(ppg, num_samples)
        # rPPG 幅值无物理单位，缩放到 [0, 1]，便于与训练 PPG 对接。
        ppg = (ppg - ppg.min()) / max(np.ptp(ppg), 1e-8)
        x = ppg[None, None, :]
        with path.with_name('summary.json').open(encoding='utf-8') as f:
            summary = json.load(f)
        timestamp = datetime.fromisoformat(summary['start_local_time'])
        timestamp += timedelta(seconds=float(wave['t_s'][0]) + args.start)
        args.timestamp = timestamp
        args.record = path.parent.name
        if args.meta is not None:
            meta = np.array([args.meta])
        else:
            if args.age is None or args.gender is None:
                raise ValueError('hwd 请传入 --age 和 --gender（或直接使用 --meta）。')
            if not np.isfinite(args.age):
                raise ValueError('年龄必须为有限数。')
            # 与 Dt.standardize_meta 一致，使用整数小时及原训练预处理的年龄范围。
            phase = 2 * np.pi * timestamp.hour / 24
            age_norm = (args.age - age_min) / (age_max - age_min)
            meta = np.array([[np.sin(phase), np.cos(phase), age_norm, args.gender]])
        target = None  # 没有同步的目标信号真值，不计算重建误差。
        print(f'Record: {path.parent.name}, window: {args.start:g}-{end:g} s')
    else:
        data = np.load(data_dir / 'Test_data.npy', mmap_mode='r')
        if not 0 <= args.sample_index < len(data):
            raise IndexError(f'共 {len(data)} 个样本，sample-index 超出范围。')
        x = np.array(data[args.sample_index:args.sample_index + 1], copy=True)
        meta = np.load(data_dir / 'Test_meta.npy', mmap_mode='r')
        meta = np.array(meta[args.sample_index:args.sample_index + 1], copy=True)
        if x.ndim != 3 or x.shape[1] != 3 or meta.shape != (1, 4):
            raise ValueError('Test_data 应为 (N, 3, L)，Test_meta 应为 (N, 4)。')
        if args.meta is not None:
            meta = np.array([args.meta])
        target_indices = [0 if name == 'ECG' else 2 for name in TARGET_MODALITIES[args.target_modality]]
        target = x[:, target_indices, :].copy()
        x = x[:, [1] if args.known_modality == 'PPG' else [0, 1], :]
    args.input_age = age_min + float(meta[0, 2]) * (age_max - age_min)
    if args.task == 'hwd' and args.meta is None:
        args.input_age = args.age
    args.age_clipped = bool(np.any((meta[:, 2] < 0) | (meta[:, 2] > 1)))
    meta[:, 2] = np.clip(meta[:, 2], 0., 1.)  # 包含 --meta 覆盖的年龄编码也限制在训练范围。
    if args.age_clipped:
        model_age = age_min + float(meta[0, 2]) * (age_max - age_min)
        print(f'Age clipped: {args.input_age:g} -> {model_age:g} (range: {age_min:g}-{age_max:g})')
    return torch.tensor(x, dtype=torch.float32), torch.tensor(meta, dtype=torch.float32), target


def infer_other_modality(model, x, meta, device, known_modality='PPG', target=None, target_modality=None):
    """保留原来的 get_embeddings → decode 推理方式。"""
    target_modality = target_modality or ('ECG' if known_modality == 'PPG' else 'ABP')
    target_names = TARGET_MODALITIES[target_modality]
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(x.to(device), known_modality, meta.to(device))
        outputs = model.decode(embeddings)
        prediction = torch.cat([outputs[0 if name == 'ECG' else 2] for name in target_names], dim=1).cpu().numpy()
    if 'ABP' in target_names:
        prediction[:, target_names.index('ABP'), :] *= 100  # 仅还原 ABP 的训练缩放。

    metrics_output = {}
    if target is not None:
        if target.shape != prediction.shape:
            raise ValueError(f'真值 {target.shape} 与重建结果 {prediction.shape} 形状不一致。')
        for index, modality in enumerate(target_names):
            original, reconstructed = target[:, index, :], prediction[:, index, :]
            metrics_output[modality] = {
                name: float(np.mean(metric(original, reconstructed)))
                for name, metric in (
                    ('mse', metrics.MSE), ('rmse', metrics.RMSE), ('mad', metrics.MAD),
                    ('prd', metrics.PRD), ('cos_sim', metrics.COS_SIM),
                    ('r_square', metrics.R_suqared), ('snr_noise', metrics.SNR),
                )
            }
        if len(target_names) == 1:
            metrics_output = metrics_output[target_names[0]]
        print(metrics_output)
    return metrics_output, target, prediction


def plot_sample(x, target, prediction, known_modality, duration, path, show=False, target_modality=None,
                sample_info='', blood_pressure=None, heart_rates=None):
    """输入与重建结果分行绘制；有真值时在同一坐标轴叠加对比。"""
    names = known_modality.split('_')
    target_modality = target_modality or ('ECG' if known_modality == 'PPG' else 'ABP')
    target_names = TARGET_MODALITIES[target_modality]
    rows = len(names) + len(target_names)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.5 * rows), sharex=True)
    sources = heart_rates['sources'] if heart_rates else {}
    for ax, name, signal in zip(axes, names, x[0].numpy()):
        ax.plot(np.arange(len(signal)) * duration / len(signal), signal, color='tab:blue')
        ax.set_ylabel(name)
        if name == 'PPG' and 'bpm' in sources.get('ppg', {}):
            hr = sources['ppg']
            peaks = np.asarray(hr['peak_indices'])
            ax.scatter(peaks * duration / len(signal), signal[peaks], color='tab:blue', s=20)
            title = f"PPG pulse rate: {hr['bpm']:.2f} bpm ({hr['n_peaks']} peaks)"
            if heart_rates['reference'] == 'original_ecg':
                error = heart_rates['difference_vs_reference_bpm']['ppg']
                title += f' | Error vs original ECG: {error:+.2f} bpm'
            ax.set_title(title, fontsize=11)
    time = np.arange(prediction.shape[-1]) * duration / prediction.shape[-1]
    for index, name in enumerate(target_names):
        ax = axes[len(names) + index]
        if target is not None:
            ax.plot(time, target[0, index], color='black', label='Original')
        ax.plot(time, prediction[0, index], color='tab:orange', label='Reconstructed')
        ax.set_ylabel(name)
        if name == 'ECG':
            hr_labels = []
            for key, waveform, color in (
                ('original_ecg', None if target is None else target[0, index], 'black'),
                ('reconstructed_ecg', prediction[0, index], 'tab:orange'),
            ):
                if waveform is not None and 'bpm' in sources.get(key, {}):
                    hr = sources[key]
                    peaks = np.asarray(hr['peak_indices'])
                    ax.scatter(time[peaks], waveform[peaks], color=color, marker='x', s=30)
                    hr_labels.append(f"{'Original' if key == 'original_ecg' else 'Reconstructed'} HR: {hr['bpm']:.2f} bpm")
            title = ' | '.join(hr_labels)
            error = heart_rates.get('difference_vs_reference_bpm', {}).get('reconstructed_ecg') if heart_rates else None
            if error is not None:
                reference = 'original ECG' if heart_rates['reference'] == 'original_ecg' else 'input PPG'
                title += f'\nReconstructed ECG - {reference}: {error:+.2f} bpm'
            ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right')
        if name == 'ABP' and blood_pressure:
            bp = blood_pressure.get('reconstructed', {})
            errors = blood_pressure.get('difference_vs_original_mmHg')
            if errors is not None:
                original_bp = blood_pressure['original']
                absolute_errors = blood_pressure['absolute_error_mmHg']
                ax.set_title('\n'.join(
                    f"{key}: Original {original_bp[key]:.2f} | Reconstructed {bp[key]:.2f} | "
                    f"Rec-Orig {errors[key]:+.2f} | Abs error {absolute_errors[key]:.2f} mmHg"
                    for key in ('SBP', 'DBP')
                ), fontsize=10)
            elif 'SBP' in bp:
                ax.set_title(f"Estimated SBP: {bp['SBP']:.1f} | DBP: {bp['DBP']:.1f} mmHg", fontsize=11)
            ax.set_ylabel('ABP (mmHg)')
    axes[rows - 1].set_xlabel('Time (s)')
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"{known_modality} → {' + '.join(target_names)}" + (f'\n{sample_info}' if sample_info else ''))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main(args):
    if args.duration <= 0:
        raise ValueError('duration 必须大于 0。')
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    args.task = args.task or config['task']
    args.target_modality = args.target_modality or ('ECG' if args.known_modality == 'PPG' else 'ABP')
    # hwd 默认复用训练数据集的权重，输出单独保存。
    checkpoint = Path(args.checkpoint or f"checkpoints/{config['task'] if args.task == 'hwd' else args.task}/model.pth")
    foldername = Path(args.output_dir or f'output/{args.task}/single_sample')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(42)  # 固定 VAE 采样，便于调整参数后比较。

    x, meta, target = load_single_sample(args, config['resample']['num_samples'])
    print('Input:', tuple(x.shape), 'Metadata:', meta.tolist())
    model_config = config['model']
    model = TripleVAE_cmeta3_fft(
        device=device,
        z_dim=model_config['z_dim'],
        latent_dim_conv=model_config['latent_dim_conv'],
        latent_dim_lstm=model_config['latent_dim_lstm'],
        in_channels=1,
        out_channels=1,
        out_features=x.shape[-1],
        merge=model_config['merge'],
        fixed=model_config['fixed'],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, target, prediction = infer_other_modality(
        model, x, meta, device, args.known_modality, target, args.target_modality,
    )

    foldername.mkdir(parents=True, exist_ok=True)
    target_name = args.target_modality.lower()
    # 从实际送入模型的元数据还原年龄和性别，兼容 --meta 覆盖。
    age_min, age_max = args.age_range
    age = age_min + float(meta[0, 2]) * (age_max - age_min)
    age_label, age_tag = f'{age:g}', f'{age:g}'
    if args.age_clipped:
        age_label, age_tag = f'{args.input_age:g} (model: {age:g})', f'{args.input_age:g}_model{age:g}'
    gender = {0: 'Female', 1: 'Male'}.get(float(meta[0, 3]), 'Unknown')
    if args.timestamp is not None:
        time_label = args.timestamp.isoformat(sep=' ', timespec='milliseconds')
        time_tag = args.timestamp.strftime('%Y%m%dT%H%M%S_%f%z')
    else:
        # II_Pleth_ABP 的保存元数据只有小时，无法还原完整日期。
        hour = int(round(np.arctan2(float(meta[0, 0]), float(meta[0, 1])) * 12 / np.pi)) % 24
        time_label, time_tag = f'{hour:02d}:00 (hour only)', f'hour{hour:02d}'
    sample_info = f'Age: {age_label} | Sex: {gender} | Window start: {time_label}'
    sample_name = f'{args.record or args.sample_index}_{args.start:g}s_{args.duration:g}s'
    prefix = foldername / f'{sample_name}_{args.known_modality}_to_{target_name}_age{age_tag}_{gender}_{time_tag}'
    for index, name in enumerate(TARGET_MODALITIES[args.target_modality]):
        output_prefix = foldername / f'{sample_name}_{args.known_modality}_to_{name.lower()}_age{age_tag}_{gender}_{time_tag}'
        np.save(f'{output_prefix}.npy', prediction[:, index:index + 1, :])
        print(f'Saved: {output_prefix}.npy')
    blood_pressure = None
    target_names = TARGET_MODALITIES[args.target_modality]
    if 'ABP' in target_names:
        fs = prediction.shape[-1] / args.duration
        blood_pressure = {'method': 'bishop', 'sampling_rate_hz': fs, 'unit': 'mmHg'}
        abp_index = target_names.index('ABP')
        for label, waveforms in (('reconstructed', prediction), ('original', target)):
            if waveforms is None:
                continue
            try:
                bp = caculate_abp_phenotype(waveforms[0, abp_index], fs)
                print(f"{label} ABP: SBP={bp['SBP']:.1f}, DBP={bp['DBP']:.1f} mmHg")
            except ValueError as exc:
                bp = {'error': str(exc)}
                print(f'{label} ABP: {exc}')
            blood_pressure[label] = bp
        if all(key in blood_pressure.get(label, {})
               for label in ('reconstructed', 'original') for key in ('SBP', 'DBP')):
            differences = {key: blood_pressure['reconstructed'][key] - blood_pressure['original'][key]
                           for key in ('SBP', 'DBP')}
            blood_pressure['error_definition'] = 'reconstructed - original'
            blood_pressure['difference_vs_original_mmHg'] = differences
            blood_pressure['absolute_error_mmHg'] = {key: abs(value) for key, value in differences.items()}
            for key, value in differences.items():
                print(f'{key} error (reconstructed - original): {value:+.2f} mmHg; absolute error: {abs(value):.2f} mmHg')
    heart_rates = None
    if 'ECG' in target_names:
        fs = prediction.shape[-1] / args.duration
        ecg_index = target_names.index('ECG')
        signals = {
            'ppg': ('PPG', x[0, args.known_modality.split('_').index('PPG')].numpy()),
            'reconstructed_ecg': ('ECG', prediction[0, ecg_index]),
        }
        if target is not None:
            signals['original_ecg'] = ('ECG', target[0, ecg_index])
        heart_rates = {'sampling_rate_hz': fs, 'definition': '60 / mean(peak_intervals_seconds)', 'sources': {}}
        for label, (modality, signal) in signals.items():
            try:
                hr = calculate_heart_rate(signal, fs, modality)
                print(f"{label} HR: {hr['bpm']:.2f} bpm ({hr['n_peaks']} peaks)")
            except ValueError as exc:
                hr = {'error': str(exc)}
                print(f'{label} HR: {exc}')
            heart_rates['sources'][label] = hr
        sources = heart_rates['sources']
        reference = next((key for key in ('original_ecg', 'ppg') if 'bpm' in sources.get(key, {})), None)
        heart_rates['reference'] = reference
        heart_rates['difference_vs_reference_bpm'] = {
            key: hr['bpm'] - sources[reference]['bpm']
            for key, hr in sources.items() if reference is not None and 'bpm' in hr
        }
        if all('bpm' in sources[key] for key in ('ppg', 'reconstructed_ecg')):
            heart_rates['reconstructed_minus_ppg_bpm'] = sources['reconstructed_ecg']['bpm'] - sources['ppg']['bpm']
    with open(f'{prefix}_stats.json', 'w', encoding='utf-8') as f:
        json.dump({'hr': heart_rates, 'bp': blood_pressure}, f, ensure_ascii=False, indent=2, allow_nan=False)
    # 合并文件保存成功后，清理同一条结果的旧版独立统计文件。
    for suffix in ('hr', 'bp'):
        Path(f'{prefix}_{suffix}.json').unlink(missing_ok=True)
    print(f'Saved: {prefix}_stats.json')
    plot_sample(x, target, prediction, args.known_modality, args.duration, f'{prefix}.png', args.show,
                target_modality=args.target_modality, sample_info=sample_info, blood_pressure=blood_pressure,
                heart_rates=heart_rates)
    print(f'Saved: {prefix}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--task', '--dataset', choices=['II_Pleth_ABP', 'hwd'], help='默认使用配置中的 task')
    parser.add_argument('--config', default='config/TriplemetaVAE3.yaml')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--checkpoint', help='模型权重路径')
    parser.add_argument('--sample-index', type=int, default=0, help='样本编号；hwd 为按目录名排序的记录编号')
    parser.add_argument('--record', help='hwd 记录目录名，例如 20261116_211644')
    parser.add_argument('--start', type=float, default=0., help='hwd 截取起点，单位秒，相对于 CSV 起点')
    parser.add_argument('--duration', type=float, default=10., help='片段时长（秒）；II_Pleth_ABP 用于时间轴和 HR/BP 采样率，不重新截取')
    parser.add_argument('--ppg-column', choices=['raw', 'filtered'], default='filtered')
    parser.add_argument('--known-modality', choices=['PPG', 'ECG_PPG'], default='PPG')
    parser.add_argument('--target-modality', choices=['ECG', 'ABP', 'both'],
                        help='重建目标；both 同时输出 ECG 和 ABP；默认 PPG → ECG、ECG_PPG → ABP')
    parser.add_argument('--age', type=float, help='hwd 实际年龄（岁），超出归一化范围时自动截断到边界')
    parser.add_argument('--gender', type=int, choices=[0, 1], help='hwd 性别：男 1 / 女 0')
    # 已按原始 II_Pleth_ABP 数据核对，并完整复现本地 Test_meta.npy。
    parser.add_argument('--age-range', nargs=2, type=float, default=(26., 91.), metavar=('MIN', 'MAX'),
                        help='年龄归一化范围；II_Pleth_ABP 默认 26 91，更换训练数据时可覆盖')
    parser.add_argument('--meta', nargs=4, type=float, metavar=('TIME_SIN', 'TIME_COS', 'AGE_NORM', 'SEX'),
                        help='直接指定编码值，优先于自动生成：时间 sin/cos、归一化年龄、性别（男 1 / 女 0）')
    parser.add_argument('--output-dir', help='默认 output/<task>/single_sample')
    parser.add_argument('--show', action='store_true', help='保存图片后显示绘图窗口')
    main(parser.parse_args())
