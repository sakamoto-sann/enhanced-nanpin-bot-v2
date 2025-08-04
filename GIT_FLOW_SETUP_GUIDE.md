# 🌊 Enhanced Nanpin Bot - Git Flow開発ワークフロー設定ガイド

## 🎯 概要

このガイドでは、Enhanced Nanpin Botプロジェクトでの効率的な開発ワークフローを確立するためのGit Flow設定について説明します。

## 🏗️ ブランチ戦略

### 📊 ブランチ構造
```
enhanced-nanpin-bot/
├── main (master)           # 🚀 本番環境 (保護ブランチ)
│   ├── tag: v1.3.0        # リリースタグ
│   ├── tag: v1.2.0        # 過去のリリース
│   └── tag: v1.1.0
│
├── develop                 # 🔧 開発統合ブランチ (保護ブランチ)
│   ├── feature/dynamic-position-sizing    # 機能開発
│   ├── feature/websocket-integration      # 機能開発
│   ├── feature/api-optimization          # 機能開発
│   └── feature/dashboard-ui               # 新機能開発
│
├── release/v1.4.0          # 🎯 リリース準備ブランチ
│
├── hotfix/auth-fix         # 🔥 緊急修正ブランチ
│
└── support/v1.3.x          # 📞 サポートブランチ（メンテナンス）
```

### 🔒 保護ブランチ設定

#### main (master) ブランチ
- **目的**: 本番環境への直接デプロイ
- **保護レベル**: 最高
- **マージ条件**:
  - Pull Requestが必須
  - レビュー承認が2名以上必要
  - 全てのステータスチェックが通過
  - ブランチが最新状態
  - 管理者でも強制プッシュ禁止

#### develop ブランチ  
- **目的**: 開発中の機能統合
- **保護レベル**: 高
- **マージ条件**:
  - Pull Requestが必須  
  - レビュー承認が1名以上必要
  - 自動テストが通過
  - コードカバレッジ80%以上

## 🚀 Git Flow設定コマンド

### 1. Git Flow初期化
```bash
# Git Flowをインストール (Linux)
sudo apt-get install git-flow

# Git Flowをインストール (macOS)
brew install git-flow

# プロジェクトでGit Flowを初期化
cd /home/tetsu/Documents/nanpin_bot
git flow init

# 設定例（推奨）
# Branch name for production releases: main
# Branch name for "next release" development: develop
# Feature branches: feature/
# Release branches: release/
# Hotfix branches: hotfix/
# Support branches: support/
# Version tag prefix: v
```

### 2. 既存ブランチの設定
```bash
# developブランチを作成（まだ存在しない場合）
git checkout -b develop
git push -u origin develop

# GitHub上でブランチ保護を設定
# Settings → Branches → Add rule で以下を設定:
# - main: 最高レベル保護
# - develop: 高レベル保護
```

## 🔄 開発ワークフロー

### 🆕 新機能開発フロー

#### Step 1: Featureブランチ作成
```bash
# 機能開発ブランチを作成
git flow feature start new-feature-name

# または手動で作成
git checkout develop
git pull origin develop
git checkout -b feature/new-feature-name
```

#### Step 2: 開発作業
```bash
# 開発作業を行う
# ファイルを編集...

# 変更をコミット
git add .
git commit -m "feat: add new feature implementation

- Implemented core functionality
- Added comprehensive tests  
- Updated documentation

🎯 Feature: new-feature-name
📊 Progress: 80% complete
🧪 Tests: All passing

Co-Authored-By: Claude <noreply@anthropic.com>"

# リモートにプッシュ
git push -u origin feature/new-feature-name
```

#### Step 3: Pull Request作成
```bash
# GitHub CLIを使用してPR作成
gh pr create \
  --title "feat: Add new feature implementation" \
  --body "$(cat <<'EOF'
## 🎯 Feature Overview
Brief description of the new feature

## 📊 Changes Made  
- [ ] Core functionality implemented
- [ ] Tests added with 80%+ coverage
- [ ] Documentation updated
- [ ] Configuration updated

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass  
- [ ] Manual testing completed

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] No breaking changes
- [ ] Ready for review

## 🔗 Related Issues
Closes #123

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)" \
  --base develop \
  --head feature/new-feature-name
```

#### Step 4: コードレビュー・マージ
```bash
# レビュー承認後、マージ
git checkout develop
git pull origin develop

# Feature完了 (Git Flow使用)
git flow feature finish new-feature-name

# または手動マージ
git merge --no-ff feature/new-feature-name
git push origin develop
git branch -d feature/new-feature-name
```

### 🚀 リリースフロー

#### Step 1: Releaseブランチ作成
```bash
# リリースブランチ作成
git flow release start v1.4.0

# または手動作成
git checkout develop
git pull origin develop
git checkout -b release/v1.4.0
```

#### Step 2: リリース準備
```bash
# バージョン更新
echo "v1.4.0" > VERSION
git add VERSION
git commit -m "chore: bump version to v1.4.0"

# CHANGELOG更新
cat >> CHANGELOG.md << 'EOF'
## [1.4.0] - 2025-08-15

### Added
- New dynamic position sizing algorithm
- Enhanced WebSocket integration
- Advanced risk management features

### Fixed  
- API authentication issues
- Memory optimization improvements

### Security
- Enhanced API key protection
EOF

git add CHANGELOG.md
git commit -m "docs: update changelog for v1.4.0"

# 最終テスト実行
npm test
npm run build
```

#### Step 3: リリース完了
```bash
# Git Flowでリリース完了
git flow release finish v1.4.0

# または手動で完了
git checkout main
git merge --no-ff release/v1.4.0
git tag -a v1.4.0 -m "Release version 1.4.0"
git checkout develop  
git merge --no-ff release/v1.4.0
git branch -d release/v1.4.0

# リモートに反映
git push origin main
git push origin develop  
git push origin --tags
```

### 🔥 Hotfix（緊急修正）フロー

#### Step 1: Hotfixブランチ作成
```bash
# 緊急修正ブランチ作成
git flow hotfix start critical-auth-fix

# または手動作成
git checkout main
git pull origin main
git checkout -b hotfix/critical-auth-fix
```

#### Step 2: 修正作業
```bash
# 修正を実装
# ファイルを修正...

git add .
git commit -m "fix: resolve critical authentication issue

🔥 Critical Fix: Authentication token validation
🎯 Impact: Prevents trading interruption  
✅ Testing: Manual verification completed
🚨 Priority: Emergency deployment required

Fixes #urgent-issue-456"

git push -u origin hotfix/critical-auth-fix
```

#### Step 3: Hotfix完了
```bash
# Git Flowで完了
git flow hotfix finish critical-auth-fix

# タグ作成とプッシュ
git push origin main
git push origin develop
git push origin --tags
```

## 🤖 GitHub Actions統合

### 📁 ワークフロー設定ファイル

#### `.github/workflows/ci.yml`
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python -m pytest tests/ --cov=src/ --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Security scan
      run: |
        pip install bandit safety
        bandit -r src/
        safety check --json
        
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t enhanced-nanpin-bot:${{ github.sha }} .
        
    - name: Deploy to staging
      if: github.ref == 'refs/heads/develop'
      run: |
        echo "Deploy to staging environment"
        
    - name: Deploy to production  
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploy to production environment"
```

#### `.github/workflows/release.yml`
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Enhanced Nanpin Bot ${{ github.ref }}
        body: |
          ## Changes in this release
          
          See [CHANGELOG.md](CHANGELOG.md) for details.
          
          ## Docker Image
          ```
          docker pull ghcr.io/sakamoto-sann/enhanced-nanpin-bot:${{ github.ref }}
          ```
```

## 🔧 設定済みGitコマンドエイリアス

### ~/.gitconfig設定
```ini
[alias]
    # Git Flow shortcuts
    fs = flow feature start
    ff = flow feature finish
    rs = flow release start
    rf = flow release finish
    hs = flow hotfix start
    hf = flow hotfix finish
    
    # 便利なコマンド
    st = status
    co = checkout
    br = branch
    ci = commit
    cm = commit -m
    cp = cherry-pick
    df = diff
    dc = diff --cached
    lg = log --oneline --graph --decorate --all
    
    # プッシュ・プル
    pom = push origin main
    pod = push origin develop
    poh = push origin HEAD
    pullom = pull origin main
    pullod = pull origin develop
    
    # ブランチ管理
    cleanup = "!git branch --merged | grep -v '\\*\\|main\\|develop' | xargs -n 1 git branch -d"
    recent = branch --sort=-committerdate
```

## 📋 チーム開発ガイドライン

### 🎯 コミットメッセージ規約

#### 基本形式
```
<type>(<scope>): <subject>

<body>

<footer>
```

#### タイプ一覧
- **feat**: 新機能
- **fix**: バグ修正  
- **docs**: ドキュメント更新
- **style**: フォーマット変更
- **refactor**: リファクタリング
- **perf**: パフォーマンス改善
- **test**: テスト追加
- **chore**: その他のタスク

#### 例
```bash
git commit -m "feat(position-sizer): implement Kelly Criterion algorithm

- Add dynamic position sizing based on Kelly Criterion
- Include risk-adjusted position calculations
- Add comprehensive unit tests
- Update configuration schema

🎯 Performance: +25% risk-adjusted returns
🧪 Coverage: 95% test coverage
📊 Impact: Core trading logic enhancement

Closes #123
Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 🔍 コードレビューガイドライン

#### レビュー観点
1. **機能性**: 要求を満たしているか
2. **品質**: コードの可読性・保守性
3. **テスト**: 適切なテストカバレッジ
4. **セキュリティ**: セキュリティ脆弱性の確認
5. **パフォーマンス**: 性能への影響
6. **ドキュメント**: 適切な文書化

#### PRテンプレート
```markdown
## 🎯 変更概要
Brief description of changes

## 📊 変更内容
- [ ] 新機能追加
- [ ] バグ修正
- [ ] リファクタリング
- [ ] ドキュメント更新

## 🧪 テスト
- [ ] 単体テスト追加・更新
- [ ] 統合テスト実行
- [ ] 手動テスト完了
- [ ] カバレッジ要件満足

## 📋 チェックリスト
- [ ] コードスタイルガイドライン遵守
- [ ] セルフレビュー完了
- [ ] 破壊的変更なし
- [ ] ドキュメント更新済み

## 🔗 関連
Fixes #issue-number
```

## 🚀 実装手順

### Phase 1: Git Flow初期設定 (今すぐ)
```bash
# 1. Git Flowインストール
brew install git-flow  # macOS
# または sudo apt-get install git-flow  # Linux

# 2. プロジェクトでの初期化
cd /home/tetsu/Documents/nanpin_bot
git flow init

# 3. developブランチ作成・プッシュ  
git checkout -b develop
git push -u origin develop
```

### Phase 2: GitHub設定 (5分)
```bash
# 1. GitHub CLIでブランチ保護設定
gh api repos/sakamoto-sann/enhanced-nanpin-bot-v2/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":2}' \
  --field restrictions=null

# 2. ワークフローファイル作成
mkdir -p .github/workflows
# 上記のci.yml, release.ymlを作成
```

### Phase 3: チーム教育 (継続)
- Git Flow ワークフローの説明
- コミットメッセージ規約の周知
- コードレビュープロセスの確立
- 自動化ツールの活用方法

## 📊 期待される効果

### 🎯 開発効率向上
- **並行開発**: 複数機能の同時開発
- **品質向上**: 体系的なレビュープロセス  
- **リスク軽減**: 段階的なリリース管理
- **自動化**: CI/CDによる作業軽減

### 🛡️ 品質・安定性向上
- **コード品質**: 必須レビューによる品質保証
- **テストカバレッジ**: 自動テストの義務化
- **セキュリティ**: 自動セキュリティスキャン
- **ドキュメント**: 変更の適切な文書化

### 🚀 運用改善
- **本番安定性**: main ブランチの厳格な保護
- **迅速な修正**: hotfix フローによる緊急対応
- **追跡可能性**: 全変更の履歴管理
- **チーム協調**: 明確な役割分担

---

**🌊 Git Flow導入により、Enhanced Nanpin Botの開発はより効率的で安全になります！**

次のステップ: Git Flow設定を実行し、最初のfeatureブランチで開発を開始しましょう。