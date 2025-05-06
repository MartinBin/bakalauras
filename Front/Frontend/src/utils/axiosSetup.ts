import axios from 'axios'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

api.defaults.withCredentials = true

api.interceptors.request.use(
  async config => {
    config.withCredentials = true

    return config
  },
  error => Promise.reject(error),
)

api.interceptors.response.use(
  response => response,
  async error => {
    const authStore = useAuthStore()
    const router = useRouter()

    if (error.response.status === 401) {
      try {
        await authStore.refreshTokenRequest()

        return api.request({ ...error.config, withCredentials: true })
      }
      catch (refreshError) {
        authStore.logout()

        router.push({ name: 'login' })
      }
    }

    return Promise.reject(error)
  },
)

export default api
