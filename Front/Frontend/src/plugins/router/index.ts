import type { App } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { routes } from './routes'
import { useAuthStore } from '@/stores/authStore'
import api from '@/utils/axiosSetup'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  if (to.meta.requiresAuth && !authStore.user)
    try {
      authStore.fetchUser()
      next()
    } catch (err) {
      next('/login')
    }
  else
    next()
})

export default function (app: App) {
  app.use(router)
}

export { router }
