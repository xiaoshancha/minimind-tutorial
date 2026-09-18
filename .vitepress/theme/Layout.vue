<script setup>
import DefaultTheme from 'vitepress/theme'
import RouteLoadingBar from './components/RouteLoadingBar.vue'
import { useRouteLoading } from './composables/routeLoading'

const { Layout } = DefaultTheme
const { loading } = useRouteLoading()
</script>

<template>
  <Layout>
    <template #layout-top>
      <RouteLoadingBar :active="loading" />
    </template>
  </Layout>
</template>
