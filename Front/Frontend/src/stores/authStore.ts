import { defineStore } from 'pinia'
import axios from 'axios'

const API_URL = 'http://localhost:8000/api/auth'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null as any | null,
  }),

  actions: {
    async login(email: string, password: string) {
      try {
        await axios.post(`${API_URL}/login/`, { email, password }, {withCredentials:true})

        await this.fetchUser()
      }
      catch (error) {
        console.error('Login failed.', error)
        throw error
      }
    },

    async fetchUser() {
      try {
        const response = await axios.get(`${API_URL}/user/`, {
          withCredentials:true
        })

        this.user = response.data
      }
      catch (error) {
        console.error(error)
      }
    },

    async refreshTokenRequest() {
      try {
        await axios.post(`${API_URL}/refresh/`, {}, {
          withCredentials:true
        })
      }
      catch (error) {
        console.error('Failed to refresh token', error)
        this.logout()
      }
    },

    logout() {
      this.user = null
      axios.post(`${API_URL}/logout/`, {}, {withCredentials:true}).catch(err => console.error(err))
    },
  },
})
