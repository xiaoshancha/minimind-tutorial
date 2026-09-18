# MiniMind 教程网站设计

- 日期：2026-09-18
- 仓库：`xiaoshancha/minimind-tutorial`（就地加 VitePress）
- 线上：`https://tutorial.baimuyuan.online/minimind-tutorial/`
- 参考：`xiaoshancha/deepseek-harness-tutorial`（VitePress 1.6 + 阿里云 Nginx 子目录）

## 目标

把现有 14 章 + 2 附录 Markdown 教材做成与 DeepSeek Harness 教程并列的静态文档站，不改写正文。

## 架构

- VitePress 1.6，源目录为仓库根
- `base: '/minimind-tutorial/'`
- 章节移到 `chapters/`，附录改为 `appendix-a-pytorch.md` / `appendix-b-glossary.md`
- 首页 `index.md` `@include` `README.md`
- `docs/`、`README.md`、`DEPLOY.md` 不进站点
- 开启 `markdown.math`（MathJax），正文大量公式
- 复用参考站路由加载条与中文 UI
- 构建产物 `.vitepress/dist/` 上传到 `/www/wwwroot/tutorial.baimuyuan.online/minimind-tutorial/`

## 信息架构

- 顶栏：首页 · 正文 · 附录
- 侧栏：开始 / 预备(1–2) / 主线(3–7) / 进阶(8–12) / 实战(13–14) / 附录
- 本地搜索、深色模式、上一章/下一章、本页目录 h2–h3

## 部署

- Cloudflare 反代 → 阿里云 ECS Nginx
- GitHub Actions：push `main` 后 build + SCP
- 本机 `deploy-local.sh` 走同一目录
- 不改域名根门户页（在另一仓库）
- checkout `fetch-depth: 0`，保证 lastUpdated

## 验收

1. `npm run build` 成功
2. dist 含首页、14 章、2 附录、带 `/minimind-tutorial/` 的资源路径、公式页含 MathJax 标记
3. 线上 `https://tutorial.baimuyuan.online/minimind-tutorial/` 200，原 `/deepseek-harness-tutorial/` 不受影响
