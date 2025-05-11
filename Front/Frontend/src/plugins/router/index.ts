import type { App } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import { routes } from './routes'
import { useAuthStore } from '@/stores/authStore'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()

  if (to.meta.requiresAuth) {
    try {
      await authStore.fetchUser()
      if (authStore.user) {
        next()
      } else {
        next({ path: '/login', replace: true })
      }
    } catch (err) {
      next({ path: '/login', replace: true })
    }
  } else {
    next()
  }
})

export default function (app: App) {
  app.use(router)
}

export { router }
