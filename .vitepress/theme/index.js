import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import { useRouteLoading } from './composables/routeLoading'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ router }) {
    const { show, hide } = useRouteLoading()
    router.onBeforeRouteChange = () => show()
    router.onAfterRouteChanged = () => hide()
  },
}
