import { readonly, ref } from 'vue'

const visible = ref(false)

let showTimer = null
let hideTimer = null
let shownAt = 0

const SHOW_DELAY = 160
const MIN_VISIBLE = 260

function show() {
  if (hideTimer) {
    clearTimeout(hideTimer)
    hideTimer = null
  }
  if (showTimer) return
  showTimer = setTimeout(() => {
    showTimer = null
    visible.value = true
    shownAt = Date.now()
  }, SHOW_DELAY)
}

function hide() {
  if (showTimer) {
    clearTimeout(showTimer)
    showTimer = null
    return
  }
  if (!visible.value) return
  if (hideTimer) clearTimeout(hideTimer)
  const elapsed = Date.now() - shownAt
  const remaining = Math.max(0, MIN_VISIBLE - elapsed)
  hideTimer = setTimeout(() => {
    hideTimer = null
    visible.value = false
  }, remaining)
}

export function useRouteLoading() {
  return { loading: readonly(visible), show, hide }
}
