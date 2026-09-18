# 部署说明：把 MiniMind 教程发布到网站

本目录已用 **VitePress** 把全部 Markdown 文档构建成一个**分层静态站**，构建产物在 `.vitepress/dist/`。它是纯静态文件（HTML/CSS/JS），只需把里面所有文件上传到任意静态服务器即可。

### 本项目已在用的实际部署

- **域名**：`tutorial.baimuyuan.online`（A 记录在 Cloudflare，橙色云 Proxied）
- **服务器**：阿里云 ECS `123.56.2.125`，Nginx 1.18
- **SSH 登录用户**：`admin`（`root` 被禁 SSH；`admin` 有免密 `sudo`），密钥 `~/.ssh/id_ed25519_penguin`
- **子路径**：`/minimind-tutorial/`（`config.mjs` 里 `base` 已设为该值）
- **网站根目录**：`/www/wwwroot/tutorial.baimuyuan.online`，内容落在其下的 `minimind-tutorial/`
- **自动部署**：GitHub Actions 工作流 `.github/workflows/deploy.yml`（push 到 main 触发）
- **本地手动部署**：`bash deploy-local.sh`

> 首次用工作流前，需在 GitHub 仓库 `Settings → Secrets and variables → Actions` 里添加：
> `SSH_HOST=123.56.2.125`、`SSH_USER=admin`、
> `SSH_PRIVATE_KEY=<id_ed25519_penguin 私钥完整内容>`。
>
> 可与 `deepseek-harness-tutorial` 仓库复用同一组 SSH secrets（端口已写死为 22）。远程路径已写死在 workflow 中，不必再配 `SSH_DEPLOY_PATH`。

---

## 1. 重新构建

```bash
npm install      # 首次
npm run build    # 输出到 .vitepress/dist/
npm run verify   # 检查产物结构、base 路径、公式标记
```

本地预览（注意：preview 会带上 base 路径）：

```bash
npm run preview   # 打开 http://localhost:4173/minimind-tutorial/
```

---

## 2. 部署

把 `.vitepress/dist/` 里的**所有内容**上传到 `/www/wwwroot/tutorial.baimuyuan.online/minimind-tutorial/`。

若 `cleanUrls` 下不带 `.html` 的章节 URL 404，给 Nginx 补：

```nginx
location /minimind-tutorial/ {
    try_files $uri $uri.html $uri/ /minimind-tutorial/index.html;
}
```

---

## 3. 常见问题

- **页面样式/JS 全 404**：多半是 `base` 配错。子路径部署必须是 `base: '/minimind-tutorial/'`，并重新 build。
- **公式没有渲染**：确认已安装 `markdown-it-mathjax3`，且 `markdown.math: true`。
- **改了文档没生效**：重新 `npm run build`，确保上传的是新的 `dist`。
