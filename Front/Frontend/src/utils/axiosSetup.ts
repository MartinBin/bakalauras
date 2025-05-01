import axios from 'axios'
import { useAuthStore } from '@/stores/authStore'
import { useRouter } from 'vue-router'

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
})

api.interceptors.request.use(
  async config => {
    const authStore = useAuthStore()
    if (authStore.accessToken)
      config.headers.Authorization = `Bearer ${authStore.accessToken}`

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
        error.config.headers.Authorization = `Bearer ${authStore.accessToken}`

        return api.request(error.config)
      }
      catch (refreshError) {
        authStore.logout()

        router.push({name: 'login'})
      }
    }

    return Promise.reject(error)
  },
)

export default api
