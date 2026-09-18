import { defineConfig } from 'vitepress'

const base = '/minimind-tutorial/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  srcExclude: ['README.md', 'DEPLOY.md', 'docs/**'],
  title: 'MiniMind 系统学习教程',
  description: '从 0 训练一个 LLM：基于 MiniMind 的教材级系统学习教程',
  cleanUrls: true,
  lastUpdated: true,

  markdown: {
    html: false,
    math: true,
    config(md) {
      md.options.html = false
      const defaultRender = md.renderer.render.bind(md.renderer)
      md.renderer.render = (tokens, options, env) => {
        const html = defaultRender(tokens, options, env)
        return html
          .replace(/\{\{/g, '&#123;&#123;')
          .replace(/\}\}/g, '&#125;&#125;')
      }
    },
  },

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
    ['meta', { name: 'theme-color', content: '#3e63dd' }],
    ['meta', { name: 'author', content: 'xiaoshancha' }],
  ],

  themeConfig: {
    logo: null,

    nav: [
      { text: '首页', link: '/' },
      { text: '正文', link: '/chapters/01-LLM全景与项目导览' },
      { text: '附录', link: '/appendix-a-pytorch' },
    ],

    sidebar: [
      {
        text: '开始',
        items: [
          { text: '书籍首页', link: '/' },
          { text: '学习路线与目录', link: '/' },
        ],
      },
      {
        text: '预备',
        collapsed: false,
        items: [
          { text: '01 · LLM 全景与项目导览', link: '/chapters/01-LLM全景与项目导览' },
          { text: '02 · 必备基础回顾', link: '/chapters/02-必备基础回顾' },
        ],
      },
      {
        text: '主线（必学）',
        collapsed: false,
        items: [
          { text: '03 · Tokenizer 与数据处理', link: '/chapters/03-Tokenizer与数据处理' },
          { text: '04 · 模型架构逐行精读', link: '/chapters/04-模型架构逐行精读' },
          { text: '05 · 预训练', link: '/chapters/05-预训练' },
          { text: '06 · SFT 指令微调', link: '/chapters/06-SFT指令微调' },
          { text: '07 · 推理与生成', link: '/chapters/07-推理与生成' },
        ],
      },
      {
        text: '进阶（可选）',
        collapsed: false,
        items: [
          { text: '08 · LoRA 参数高效微调', link: '/chapters/08-LoRA参数高效微调' },
          { text: '09 · RLHF-DPO 偏好对齐', link: '/chapters/09-RLHF-DPO偏好对齐' },
          { text: '10 · RLAIF：PPO 与 GRPO', link: '/chapters/10-RLAIF-PPO与GRPO' },
          { text: '11 · Agentic RL 与 Tool Call', link: '/chapters/11-AgenticRL与ToolCall' },
          { text: '12 · 知识蒸馏', link: '/chapters/12-知识蒸馏' },
        ],
      },
      {
        text: '实战',
        collapsed: false,
        items: [
          { text: '13 · 从 0 到 1 完整实战', link: '/chapters/13-从0到1完整实战' },
          { text: '14 · 项目扩展与进阶路线', link: '/chapters/14-项目扩展与进阶路线' },
        ],
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: 'A · PyTorch 速查', link: '/appendix-a-pytorch' },
          { text: 'B · LLM 术语表', link: '/appendix-b-glossary' },
        ],
      },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/xiaoshancha/minimind-tutorial' },
    ],

    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一章', next: '下一章' },
    lastUpdated: { text: '最后更新', formatOptions: { dateStyle: 'short', timeStyle: 'short' } },
    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式',
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '未找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭',
            },
          },
        },
      },
    },
  },
})
