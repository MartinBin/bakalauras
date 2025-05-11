import { vi, beforeAll, afterAll, afterEach } from 'vitest'
import axios from 'axios'
import { vuetify } from './vuetify'
import { config } from '@vue/test-utils'

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

global.ResizeObserver = ResizeObserverMock

config.global.plugins = [vuetify]

beforeAll(() => {
  vi.mock('axios', () => ({
    default: {
      post: vi.fn(),
      get: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
      create: vi.fn(() => ({
        post: vi.fn(),
        get: vi.fn(),
        put: vi.fn(),
        delete: vi.fn(),
      })),
    },
  }))
})

afterEach(() => {
  vi.clearAllMocks()
})

afterAll(() => {
  vi.resetAllMocks()
})

vi.mock('*.svg', () => ({
  default: '<svg></svg>',
}))

vi.mock('*.svg?raw', () => ({
  default: '<svg></svg>',
}))

export const mockAxios = {
  post: vi.fn(),
  get: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
  create: vi.fn(() => ({
    post: mockAxios.post,
    get: mockAxios.get,
    put: mockAxios.put,
    delete: mockAxios.delete,
  })),
}

vi.mock('axios', () => ({
  default: mockAxios,
  ...mockAxios,
}))

vi.mock('*.css', () => ({}))
vi.mock('*.scss', () => ({}))
vi.mock('*.sass', () => ({}))
vi.mock('*.less', () => ({}))
vi.mock('*.styl', () => ({}))
vi.mock('*.stylus', () => ({}))
vi.mock('*.pcss', () => ({}))
vi.mock('*.postcss', () => ({}))
vi.mock('*.sss', () => ({}))

const originalConsoleError = console.error

console.error = vi.fn()

afterAll(() => {
  console.error = originalConsoleError
})
