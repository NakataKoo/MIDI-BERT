# MidiBERT-Piano


事前学習済みモデルの重み＋データセット：https://huggingface.co/KooNakata/MIDI-BERT

## Introduction
This is not official repository for the paper, [MidiBERT-Piano: Large-scale Pre-training for Symbolic Music Understanding](https://arxiv.org/pdf/2107.05223.pdf).

With this repository, you can
* pre-train a MidiBERT-Piano with your customized pre-trained dataset
* fine-tune & evaluate on 4 downstream tasks
* extract melody (mid to mid) using pre-trained MidiBERT-Piano

All the datasets employed in this work are publicly available.

## Installation
* Python3.9
* Install generally used packages for MidiBERT-Piano:
```python
git clone https://github.com/wazenmai/MIDI-BERT.git
cd MIDI-BERT
pip install -r requirements.txt
```

研究室のA40サーバー（CUDA 12.1）では、以下でも上手くいった(torch=2.2.0, CUDA=12.1,)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Python3.10の場合
requirements.txtを以下の様に変更
```
numpy>=1.13.3
matplotlib>=3.3.3
mido==1.2.10
#torch>=1.3.1
chorder==0.1.2
#miditoolkit==0.1.14
#scikit_learn==0.24.2
#torchaudio==0.9.0
transformers==4.8.2
SoundFile
tqdm
pypianoroll
```
次に、以下を実行（CUDA 12.1の場合）
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

miditoolkitやscikit_learnはバージョン指定せず、個別にインストールするのでも良さそう

## Customize your own pre-training dataset 'lmd_aligned'

1. data_creation/prepare_dataのmain.py, model.py, utils.pyのimportにおいて、data_creation.prepare_data.の部分を削除
2. ```!wget http://hog.ee.columbia.edu/craffel/lmd/lmd_aligned.tar.gz```を実行
3. ```!tar -zxvf lmd_aligned.tar.gz```を実行し解凍
4. 以下でdata_creation/prepare_data/dict/CP.pklの中身を表示
```python
import pickle

# CP.pklファイルのパス
file_path = 'data_creation/prepare_data/dict/CP.pkl'

# ファイルを読み込み
with open(file_path, 'rb') as f:
    cp_dict = pickle.load(f)

# データの表示
print(cp_dict)
```
5. 以下でCP.pklにおいて、Pitchの範囲を0~127へ拡大
```python
import pickle

# 既存の辞書を読み込み
dict_path = 'data_creation/prepare_data/dict/CP.pkl'
with open(dict_path, 'rb') as f:
    event2word, word2event = pickle.load(f)

# ピッチの範囲
min_pitch = 0
max_pitch = 127

# ピッチのエントリを追加
for pitch in range(min_pitch, max_pitch + 1):
    pitch_key = f'Pitch {pitch}'
    if pitch_key not in event2word['Pitch']:
        event2word['Pitch'][pitch_key] = -1  # 一時的に-1を設定

# ピッチのキーを昇順にソートして再割り当て
special_keys = {'Pitch <PAD>', 'Pitch <MASK>'}
sorted_pitch_keys = sorted(
    [k for k in event2word['Pitch'].keys() if k not in special_keys],
    key=lambda x: int(x.split()[1])
)

# 特別なキーは元の位置に戻す
for new_index, pitch_key in enumerate(sorted_pitch_keys):
    event2word['Pitch'][pitch_key] = new_index
    word2event['Pitch'][new_index] = pitch_key

# 特別なキーを追加
current_index = len(sorted_pitch_keys)
for special_key in special_keys:
    event2word['Pitch'][special_key] = current_index
    word2event['Pitch'][current_index] = special_key
    current_index += 1

# 更新された辞書を保存
with open(dict_path, 'wb') as f:
    pickle.dump((event2word, word2event), f)

print("CP.pklを更新しました。")
```
6. utils.pyの29行目以降を以下の様に変更
```python
try:
   midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
except OSError as e:
   print(f"Error reading {file_path}: {e}")
   return [], []  # 空のリストを返してエラーを処理
```

7. main.pyの117行目あたりを、以下に変更
```python
elif args.input_dir == "lmd_aligned":
    files = glob.glob('lmd_aligned/**/*.mid', recursive=True)
```

8. ルードディレクトリにて以下を実行し、データセットに存在しないファイル＋サブディレクトリをlmd_alignedフォルダから削除
```python
import pandas as pd
import os
import shutil

# CSVファイルを読み込む
df = pd.read_csv('/content/midi_mp3_caption_clean.csv')

# 「lmd_aligned」列に存在するフォルダ名のリストを取得
existing_folders = df['lmd_aligned'].tolist()

# ディレクトリAのパスを指定
directory_a = 'lmd_aligned/'

# ディレクトリA内の一番下の階層のみを走査
for root, dirs, files in os.walk(directory_a):
    if not dirs:  # サブディレクトリがない、つまり一番下の階層である場合
        if root not in existing_folders:
            # 一番下のフォルダが「lmd_aligned」列に存在しない場合、そのフォルダを削除
            shutil.rmtree(root)
            print(f"Deleted folder: {root}")

def remove_empty_dirs(directory):
    # ディレクトリ内を再帰的に走査
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # ディレクトリが空かどうかを確認
            if not os.listdir(dir_path):
                # 空のディレクトリを削除
                os.rmdir(dir_path)
                print(f"Deleted empty directory: {dir_path}")

# 対象のディレクトリを指定
directory = '/content/MIDI-BERT/lmd_aligned'

# 空のサブディレクトリを削除
remove_empty_dirs(directory)
```

以下で、1,7077となることを確認
```find lmd_aligned -type d -links 2 | wc -l```

9. 以下でMIDI-BERT入力用データの前処理実行
```
input_dir="lmd_aligned"
!export PYTHONPATH='.'

# custom directory
!python3 data_creation/prepare_data/main.py --input_dir=$input_dir --name="lmd_aligned"
```

## Citation

[Midi-BERT Official repo](https://github.com/wazenmai/MIDI-BERT/tree/CP).
