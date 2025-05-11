import { describe, it, expect, beforeEach, vi } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import { useAuthStore } from '../stores/authStore'
import axios from 'axios'

vi.mock('axios')
const mockedAxios = vi.mocked(axios)

describe('Auth Store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  describe('login', () => {
    it('should successfully login user', async () => {
      const store = useAuthStore()
      const mockUser = { id: 1, email: 'test@example.com' }

      vi.mocked(mockedAxios.post).mockResolvedValueOnce({ data: {} })
      vi.mocked(mockedAxios.get).mockResolvedValueOnce({ data: mockUser })

      await store.login('test@example.com', 'password')

      expect(mockedAxios.post).toHaveBeenCalledWith(
        'http://localhost:8000/api/auth/login/',
        { email: 'test@example.com', password: 'password' },
        { withCredentials: true },
      )
      expect(store.user).toEqual(mockUser)
    })

    it('should throw error on login failure', async () => {
      const store = useAuthStore()
      const error = new Error('Login failed')

      vi.mocked(mockedAxios.post).mockRejectedValueOnce(error)

      await expect(store.login('test@example.com', 'wrong-password')).rejects.toThrow()
      expect(store.user).toBeNull()
    })
  })

  describe('fetchUser', () => {
    it('should fetch user data successfully', async () => {
      const store = useAuthStore()
      const mockUser = { id: 1, email: 'test@example.com' }

      vi.mocked(mockedAxios.get).mockResolvedValueOnce({ data: mockUser })

      await store.fetchUser()

      expect(vi.mocked(mockedAxios.get)).toHaveBeenCalledWith('http://localhost:8000/api/auth/user/', {
        withCredentials: true,
      })
      expect(store.user).toEqual(mockUser)
    })

    it('should handle fetch user error', async () => {
      const store = useAuthStore()
      const error = new Error('Failed to fetch user')

      vi.mocked(mockedAxios.get).mockRejectedValueOnce(error)

      await store.fetchUser()

      expect(store.user).toBeNull()
    })
  })

  describe('refreshToken', () => {
    it('should refresh token successfully', async () => {
      const store = useAuthStore()

      vi.mocked(mockedAxios.post).mockResolvedValueOnce({ data: {} })

      await store.refreshTokenRequest()

      expect(vi.mocked(mockedAxios.post)).toHaveBeenCalledWith(
        'http://localhost:8000/api/auth/refresh/',
        {},
        { withCredentials: true },
      )
    })

    it('should logout on refresh token failure', async () => {
      const store = useAuthStore()
      const error = new Error('Token refresh failed')

      vi.mocked(mockedAxios.post).mockRejectedValueOnce(error)

      await store.refreshTokenRequest()

      expect(store.user).toBeNull()
      expect(mockedAxios.post).toHaveBeenCalledWith(
        'http://localhost:8000/api/auth/logout/',
        {},
        { withCredentials: true },
      )
    })
  })

  describe('logout', () => {
    it('should clear user data and call logout endpoint', async () => {
      const store = useAuthStore()
      store.user = { id: 1, email: 'test@example.com' }

      vi.mocked(mockedAxios.post).mockResolvedValueOnce({ data: {} })

      await store.logout()

      expect(store.user).toBeNull()
      expect(mockedAxios.post).toHaveBeenCalledWith(
        'http://localhost:8000/api/auth/logout/',
        {},
        { withCredentials: true },
      )
    })

    it('should handle logout error gracefully', async () => {
      const store = useAuthStore()
      store.user = { id: 1, email: 'test@example.com' }

      vi.mocked(mockedAxios.post).mockRejectedValueOnce(new Error('Logout failed'))

      await store.logout()

      expect(store.user).toBeNull()
    })
  })
})
