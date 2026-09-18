<script setup>
defineProps({
  active: { type: Boolean, default: false },
})
</script>

<template>
  <Teleport to="body">
    <div class="route-loading" :class="{ 'is-active': active }" aria-hidden="true">
      <div class="route-loading__spinner" />
    </div>
  </Teleport>
</template>

<style scoped>
.route-loading {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2147483000;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.22s ease;
}

.route-loading.is-active {
  opacity: 1;
}

.route-loading__spinner {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  border: 4px solid rgba(62, 99, 221, 0.18);
  border-top-color: #3e63dd;
  animation: route-loading-spin 0.85s linear infinite;
}

@keyframes route-loading-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
