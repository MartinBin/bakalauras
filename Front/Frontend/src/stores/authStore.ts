import { defineStore } from 'pinia'
import axios from 'axios'

const API_URL = 'http://localhost:8080/api/auth/'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null as any | null,
    accessToken: localStorage.getItem('accessToken') || null,
    refreshToken: localStorage.getItem('refreshToken') || null,
  }),

  actions: {
    async login(email: string, password: string) {
      try {
        const response = await axios.post(`${API_URL}/login`, { email, password })

        this.accessToken = response.data.accessToken
        this.refreshToken = response.data.refreshToken
        localStorage.setItem('accessToken', response.data.accessToken)
        localStorage.setItem('refreshToken', response.data.refreshToken)
        axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.accessToken}`
        await this.fetchUser()

        return response.data
      }
      catch (error) {
        console.error('Login failed.', error)
        throw error
      }
    },

    async fetchUser() {
      try {
        const response = await axios.get(`${API_URL}/user`)

        this.user = response.data
      }
      catch (error) {
        console.error(error)
      }
    },

    async refreshTokenRequest() {
      try {
        const response = await axios.post(`${API_URL}token/refresh/`, {
          refresh: this.refreshToken,
        })

        this.accessToken = response.data.access
        localStorage.setItem('access_token', this.accessToken)
        axios.defaults.headers.common['Authorization'] = `Bearer ${this.accessToken}`
      }
      catch (error) {
        console.error('Failed to refresh token', error)
        this.logout()
      }
    },

    logout() {
      this.user = null
      this.accessToken = null
      this.refreshToken = null
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      delete axios.defaults.headers.common['Authorization']

      axios.post(`${API_URL}logout/`, { refresh: this.refreshToken }).catch(err => console.error(err))
    },
  },
})
