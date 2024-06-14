# Genifyer & Clairvoyant
データの拡充と意思決定の AI. 2024年6月から.

## 概要
2種類の AI

- データの拡充 "Genifyer"
- 意思決定 "Clairvoyant"

の融合システムによる, 「少ないデータから最良意思決定をする」というテーマに沿ったプロジェクト.  

### Genifyer
Generate + Amplify + er の造語.  
VAE(変分オートエンコーダ)をベースにしたデータ拡充用 AI モデル.

### Clairvoyant
反実仮想機械学習の1種であるオフ方策評価を用いた意思決定促進 AI モデル.

## フォルダの説明


## ローカル環境のセットアップ
`PYTHONPATH` を通して追加する.

```
echo 'export PYTHONPATH=../..' >> ~/.bashrc
source ~/.bashrc
```

仮想環境を構築する.

```
python3 -m venv Clairvoyant
source ./clairvoyant/bin/activate
```

`pip3` を使用する場合, リポジトリのターミナル上で以下のコマンドを実行する.

```
pip3 install --upgrade pip
pip3 install -r requirements.txt
```


## 実行方法


## Commit ルール
Commit の際は以下のルールに合わせて種類ごとにする.

🎉  初めてのコミット (Initial Commit)  
🔖  バージョンタグ (Version Tag)  
✨  新機能 (New Feature)  
🐛  バグ修正 (Bugfix)  
♻️  リファクタリング (Refactoring)  
📚  ドキュメント (Documentation)  
🎨  デザインUI/UX (Accessibility)  
🐎  パフォーマンス (Performance)  
🔧  ツール (Tooling)  
🚨  テスト (Tests)  
💩  非推奨追加 (Deprecation)  
🗑️  削除 (Removal)  
🚧  WIP (Work In Progress)