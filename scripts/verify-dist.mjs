import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const dist = '.vitepress/dist'
const errors = []

function fail(message) {
  errors.push(message)
}

function mustExist(rel) {
  if (!existsSync(join(dist, rel))) fail(`missing ${rel}`)
}

if (!existsSync(dist)) {
  console.error('dist 不存在，请先 npm run build')
  process.exit(1)
}

mustExist('index.html')
mustExist('404.html')
mustExist('favicon.svg')
mustExist('appendix-a-pytorch.html')
mustExist('appendix-b-glossary.html')

const chapterDir = join(dist, 'chapters')
if (!existsSync(chapterDir)) {
  fail('missing chapters/')
} else {
  const html = readdirSync(chapterDir).filter((name) => name.endsWith('.html'))
  if (html.length !== 14) {
    fail(`expected 14 chapter html files, got ${html.length}: ${html.join(', ')}`)
  }
}

const index = readFileSync(join(dist, 'index.html'), 'utf8')
if (!index.includes('/minimind-tutorial/assets/')) {
  fail('index.html 缺少 /minimind-tutorial/assets/，base 可能配错')
}
if (!index.includes('MiniMind')) {
  fail('index.html 未包含 MiniMind')
}

const distFiles = []
function walk(dir) {
  for (const name of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, name.name)
    if (name.isDirectory()) walk(full)
    else distFiles.push(full)
  }
}
walk(dist)

const blob = distFiles
  .filter((file) => /\.(html|js)$/.test(file))
  .map((file) => readFileSync(file, 'utf8'))
  .join('\n')

if (!blob.includes('01 · LLM 全景与项目导览') && !blob.includes('LLM 全景与项目导览')) {
  fail('构建产物中找不到第 1 章侧栏/标题')
}

const hasMath =
  blob.includes('mjx-') ||
  blob.includes('MathJax') ||
  blob.includes('katex') ||
  blob.includes('mathjax') ||
  blob.includes('\\frac') ||
  blob.includes('class="math')
if (!hasMath) {
  fail('构建产物中找不到公式标记（MathJax/KaTeX）')
}

if (errors.length) {
  console.error('verify-dist 失败：')
  for (const error of errors) console.error(`- ${error}`)
  process.exit(1)
}

console.log('verify-dist 通过：首页、14 章、2 附录、base 路径、公式标记均在')
