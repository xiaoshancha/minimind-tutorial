# MiniMind 教程网站 Implementation Plan

> **For agentic workers:** 本计划在当前会话内联执行。证据路径是 `npm run build` + `node scripts/verify-dist.mjs`，不是单元测试。

**Goal:** 把 `minimind-tutorial` 做成 VitePress 静态站，部署到 `https://tutorial.baimuyuan.online/minimind-tutorial/`。

**Architecture:** 仓库根为 VitePress 源；章节进 `chapters/`；`base` 为 `/minimind-tutorial/`；产物 SCP 到阿里云 Nginx 子目录。

**Tech Stack:** VitePress ^1.6.3，markdown-it-mathjax3，GitHub Actions + appleboy/scp-action。

## Global Constraints

- 不改写章节正文，只搬家和加脚手架
- `base` 必须是 `/minimind-tutorial/`
- 公式必须能渲染
- 不碰 `/deepseek-harness-tutorial/`
- 本机可能没有 ECS SSH 密钥；Actions secrets 需仓库维护者配置

---

### Task 1: 站点脚手架与内容搬家

**Files:**
- Create: `package.json`, `.gitignore`, `.gitattributes`, `index.md`, `.vitepress/config.mjs`, `.vitepress/theme/*`, `public/favicon.svg`, `DEPLOY.md`, `deploy-local.sh`, `.github/workflows/deploy.yml`, `scripts/verify-dist.mjs`
- Move: `01-*.md`…`14-*.md` → `chapters/`；附录改英文 slug
- Modify: `README.md` 增加在线阅读链接

**Verify:** `npm ci && npm run build && node scripts/verify-dist.mjs`

---

### Task 2: 部署

- 有 SSH 密钥则 `bash deploy-local.sh`
- 否则 commit + push，依赖 GitHub Actions secrets
- 线上 curl 首页与第 4 章，确认原站仍 200
