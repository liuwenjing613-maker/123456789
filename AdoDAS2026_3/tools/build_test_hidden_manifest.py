from pathlib import Path
import pandas as pd

ROOT = Path("/home/adodas/dataset/test_hidden").resolve()
OUT = Path("/home/adodas/dataset/manifests/test_hidden.csv").resolve()

SESSIONS = {"A01", "B01", "B02", "B03"}
MODALITIES = {"audio", "video"}

def parse_sequence_path(seq_path: Path):
    """
    从 sequence.npz 路径中解析：
    anon_school, anon_class, anon_pid, session

    支持两类目录：
    1) school-first:
       SCH/CLASS/PID/audio/feature[/tag]/A01/sequence.npz

    2) feature-first:
       audio/feature[/tag]/SCH/CLASS/PID/A01/sequence.npz
       或 ORGANIZED_BY_FEATURES/audio/feature[/tag]/SCH/CLASS/PID/A01/sequence.npz
    """
    rel = seq_path.relative_to(ROOT)
    parts = rel.parts

    if len(parts) < 6:
        return None

    if parts[-1] != "sequence.npz":
        return None

    session = parts[-2]
    if session not in SESSIONS:
        return None

    # 情况 1：官方 school-first
    # SCH / CLASS / PID / audio|video / ...
    if len(parts) >= 6 and parts[3] in MODALITIES:
        sch = parts[0]
        cls = parts[1]
        pid = parts[2]
        return sch, cls, pid, session

    # 情况 2：feature-first
    # ... / audio|video / feature / [tag] / SCH / CLASS / PID / session / sequence.npz
    # 对这种结构，session 前面三个目录就是 SCH / CLASS / PID
    if len(parts) >= 5:
        sch = parts[-5]
        cls = parts[-4]
        pid = parts[-3]

        # 简单过滤，避免把 feature 名错当成 school
        bad = MODALITIES | {
            "mel_mfcc", "vad", "egemaps", "ssl_embed",
            "headpose_geom", "face_behavior", "qc_stats",
            "vad_agg", "body_pose", "global_motion", "vision_ssl_embed",
            "wav2vec2-chinese-xlsr", "dinov2-large",
        }

        if sch in bad or cls in bad or pid in bad:
            return None

        # 路径前面必须出现 audio 或 video，证明这是特征路径
        if any(x in MODALITIES for x in parts[:-5]):
            return sch, cls, pid, session

    return None

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"test_hidden directory not found: {ROOT}")

    seq_files = list(ROOT.rglob("sequence.npz"))
    print(f"[INFO] found sequence.npz files: {len(seq_files)}")

    if not seq_files:
        print("[ERROR] 没有找到 sequence.npz。")
        print("请先确认测试集特征是否解压，而不是只解压了 test_transcript.zip。")
        print("检查命令：")
        print("  find /home/adodas/dataset/test_hidden -type f | head -50")
        raise SystemExit(1)

    records = []
    bad_examples = []

    for p in seq_files:
        parsed = parse_sequence_path(p)
        if parsed is None:
            if len(bad_examples) < 10:
                bad_examples.append(str(p))
            continue

        sch, cls, pid, session = parsed
        records.append({
            "anon_school": sch,
            "anon_class": cls,
            "anon_pid": pid,
            "session": session,
        })

    if not records:
        print("[ERROR] 找到了 sequence.npz，但没有成功解析出 school/class/pid/session。")
        print("无法解析的路径示例：")
        for x in bad_examples:
            print("  ", x)
        print("请把下面命令输出发给我：")
        print("  find /home/adodas/dataset/test_hidden -type f -name sequence.npz | head -50")
        raise SystemExit(1)

    df = pd.DataFrame(records).drop_duplicates()
    df = df.sort_values(["anon_school", "anon_class", "anon_pid", "session"])

    # A1 假标签
    df["y_D"] = -1
    df["y_A"] = -1
    df["y_S"] = -1

    # A2 假标签，方便同一份 manifest 也能被 A2 代码读取
    for i in range(1, 22):
        df[f"d{i:02d}"] = -1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    n_participants = df[["anon_school", "anon_class", "anon_pid"]].drop_duplicates().shape[0]

    print("[OK] wrote:", OUT)
    print("[OK] rows:", len(df))
    print("[OK] participants:", n_participants)
    print("[OK] session counts:")
    print(df["session"].value_counts().sort_index())
    print("[OK] head:")
    print(df.head(12).to_string(index=False))

    # 检查每个 participant 是否有 4 个 session
    cnt = df.groupby(["anon_school", "anon_class", "anon_pid"])["session"].nunique()
    print("[INFO] participants with 4 sessions:", int((cnt == 4).sum()))
    print("[INFO] participants with <4 sessions:", int((cnt < 4).sum()))

if __name__ == "__main__":
    main()
