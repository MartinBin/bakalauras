import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, type VueWrapper, flushPromises } from '@vue/test-utils'
import Dashboard from '../pages/dashboard.vue'
import api from '../utils/axiosSetup'
import { createVuetify } from 'vuetify'
import type { ComponentPublicInstance } from 'vue'

const mockObjectURL = 'blob:test'
global.URL = {
  createObjectURL: vi.fn(() => mockObjectURL),
  revokeObjectURL: vi.fn(),
} as unknown as typeof global.URL

vi.mock('../components/PointCloudViewer.vue', () => ({
  default: {
    name: 'PointCloudViewer',
    template: '<div class="point-cloud-viewer"></div>',
  },
}))

vi.mock('../components/MetricsChart.vue', () => ({
  default: {
    name: 'MetricsChart',
    template: '<div class="metrics-chart"></div>',
  },
}))

const mockRouter = {
  push: vi.fn(),
}

vi.mock('vue-router', () => ({
  useRouter: () => mockRouter,
}))

interface DashboardInstance extends ComponentPublicInstance {
  leftImage: File | null
  rightImage: File | null
  leftPreview: string
  rightPreview: string
  isLoading: boolean
  error: string
  predictionResult: string
  currentMetrics: {
    variance?: number
    std_dev?: number
    confidence_score?: number
    point_count?: number
  } | null
  handleLeftImageChange: (event: Event) => void
  handleRightImageChange: (event: Event) => void
  handleSubmit: () => Promise<void>
}

vi.mock('../utils/axiosSetup', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    head: vi.fn(),
  },
}))

const vuetify = createVuetify()

describe('Dashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.head).mockResolvedValue({ status: 200 })
    vi.mocked(api.post).mockResolvedValue({
      data: {
        point_cloud_path: 'media/pointclouds/test.ply',
        metrics: {
          variance: 0.1234,
          std_dev: 0.5678,
          confidence_score: 0.95,
          point_count: 1000,
        },
        unet_outputs: {
          left: 'media/unet/left.jpg',
          right: 'media/unet/right.jpg',
        },
        depth_values: {
          left: Array(512 * 512).fill(1),
          right: Array(512 * 512).fill(1),
        },
      },
    })
  })

  it('renders properly', () => {
    const wrapper = mount(Dashboard, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
          MetricsChart: true,
        },
      },
    })

    const title = wrapper.find('.v-card-title span')
    expect(title.text()).toBe('3D Point Cloud Generation')

    const generateButton = wrapper.findAll('.v-btn').find(btn => btn.text() === 'Generate Point Cloud')
    expect(generateButton).toBeTruthy()
  })

  it('handles image file selection', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
          MetricsChart: true,
        },
      },
    }) as VueWrapper<DashboardInstance>

    const leftFile = new File(['test'], 'left.jpg', { type: 'image/jpeg' })
    const rightFile = new File(['test'], 'right.jpg', { type: 'image/jpeg' })

    const leftEvent = {
      target: {
        files: [leftFile],
      },
    } as unknown as Event

    const rightEvent = {
      target: {
        files: [rightFile],
      },
    } as unknown as Event

    await wrapper.vm.handleLeftImageChange(leftEvent)
    await wrapper.vm.handleRightImageChange(rightEvent)

    expect(wrapper.vm.leftImage).toBe(leftFile)
    expect(wrapper.vm.rightImage).toBe(rightFile)
    expect(wrapper.vm.leftPreview).toBe(mockObjectURL)
    expect(wrapper.vm.rightPreview).toBe(mockObjectURL)
  })

  it('shows error when submitting without images', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
          MetricsChart: true,
        },
      },
    }) as VueWrapper<DashboardInstance>

    await wrapper.vm.handleSubmit()
    await flushPromises()

    expect(wrapper.vm.error).toBe('Please select both left and right images')
  })

  it('handles successful point cloud generation', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
          MetricsChart: true,
        },
      },
    }) as VueWrapper<DashboardInstance>

    const leftFile = new File(['test'], 'left.jpg', { type: 'image/jpeg' })
    const rightFile = new File(['test'], 'right.jpg', { type: 'image/jpeg' })

    const leftEvent = {
      target: {
        files: [leftFile],
      },
    } as unknown as Event

    const rightEvent = {
      target: {
        files: [rightFile],
      },
    } as unknown as Event

    await wrapper.vm.handleLeftImageChange(leftEvent)
    await wrapper.vm.handleRightImageChange(rightEvent)

    await wrapper.vm.handleSubmit()
    await flushPromises()

    expect(wrapper.vm.predictionResult).toBe('http://localhost:8000/media/pointclouds/test.ply')
    expect(wrapper.vm.currentMetrics).toEqual({
      variance: 0.1234,
      std_dev: 0.5678,
      confidence_score: 0.95,
      point_count: 1000,
    })
  })

  it('handles API errors gracefully', async () => {
    vi.mocked(api.post).mockRejectedValueOnce(new Error('API Error'))

    const wrapper = mount(Dashboard, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
          MetricsChart: true,
        },
      },
    }) as VueWrapper<DashboardInstance>

    const leftFile = new File(['test'], 'left.jpg', { type: 'image/jpeg' })
    const rightFile = new File(['test'], 'right.jpg', { type: 'image/jpeg' })

    const leftEvent = {
      target: {
        files: [leftFile],
      },
    } as unknown as Event

    const rightEvent = {
      target: {
        files: [rightFile],
      },
    } as unknown as Event

    await wrapper.vm.handleLeftImageChange(leftEvent)
    await wrapper.vm.handleRightImageChange(rightEvent)

    await wrapper.vm.handleSubmit()
    await flushPromises()

    expect(wrapper.vm.error).toBe('Error processing images. Please try again.')
  })

  it('navigates to history page when clicking view history button', async () => {
    const wrapper = mount(Dashboard, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
          MetricsChart: true,
        },
      },
    })

    const historyButton = wrapper.findAll('.v-btn').find(btn => btn.text() === 'View History')
    await historyButton?.trigger('click')

    expect(mockRouter.push).toHaveBeenCalledWith('/prediction-history')
  })
})
