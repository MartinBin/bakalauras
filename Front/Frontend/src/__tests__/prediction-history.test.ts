import { describe, it, expect, vi, beforeEach } from 'vitest'
import { mount, type VueWrapper, flushPromises } from '@vue/test-utils'
import PredictionHistory from '../pages/prediction-history.vue'
import api from '../utils/axiosSetup'
import { createVuetify } from 'vuetify'
import type { ComponentPublicInstance } from 'vue'

vi.mock('../components/PointCloudViewer.vue', () => ({
  default: {
    name: 'PointCloudViewer',
    template: '<div class="point-cloud-viewer"></div>',
  },
}))

interface PredictionHistoryInstance extends ComponentPublicInstance {
  dialog: boolean
  selectedFile: string | undefined
  predictions: Array<{
    id: string
    timestamp: number
    leftImage?: string
    rightImage?: string
    predictedPointCloud?: string
    metrics: {
      variance: number
      std_dev: number
      confidence_score?: number
      point_count?: number
    } | null
  }>
}

const mockPredictions = [
  {
    id: '1',
    created_at: '2024-03-20T10:00:00Z',
    metadata: {
      left_image_path: 'images/left1.jpg',
      right_image_path: 'images/right1.jpg',
    },
    point_cloud_path: 'pointclouds/cloud1.ply',
    metrics: {
      depth_confidence: 0.95,
    },
  },
  {
    id: '2',
    created_at: '2024-03-20T11:00:00Z',
    metadata: {
      left_image_path: 'images/left2.jpg',
      right_image_path: 'images/right2.jpg',
    },
    point_cloud_path: null,
    metrics: null,
  },
]

vi.mock('../utils/axiosSetup', () => ({
  default: {
    get: vi.fn(),
    delete: vi.fn(),
  },
}))

const vuetify = createVuetify()

describe('PredictionHistory', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(api.get).mockResolvedValue({ data: mockPredictions })
  })

  it('displays "No prediction history available" when there are no predictions', async () => {
    vi.mocked(api.get).mockResolvedValue({ data: [] })

    const wrapper = mount(PredictionHistory, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
        },
      },
    })

    await flushPromises()

    expect(wrapper.text()).toContain('No prediction history available')
  })

  it('removes prediction when clicking delete button', async () => {
    vi.mocked(api.delete).mockResolvedValue({})

    const wrapper = mount(PredictionHistory, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
        },
      },
    }) as VueWrapper<PredictionHistoryInstance>

    await flushPromises()

    const buttons = wrapper.findAllComponents({ name: 'VBtn' })
    const deleteButton = buttons.find(btn => btn.text().includes('Delete'))
    expect(deleteButton?.exists()).toBe(true)

    await deleteButton?.trigger('click')

    expect(api.delete).toHaveBeenCalledWith('/user/predictions/2/')
    expect(wrapper.vm.predictions.length).toBe(1)
  })

  it('clears all predictions when clicking clear history button', async () => {
    vi.mocked(api.delete).mockResolvedValue({})

    const wrapper = mount(PredictionHistory, {
      global: {
        plugins: [vuetify],
        stubs: {
          PointCloudViewer: true,
        },
      },
    }) as VueWrapper<PredictionHistoryInstance>

    await flushPromises()

    const buttons = wrapper.findAllComponents({ name: 'VBtn' })
    const clearButton = buttons.find(btn => btn.text().includes('Clear History'))
    expect(clearButton?.exists()).toBe(true)

    await clearButton?.trigger('click')

    expect(api.delete).toHaveBeenCalledWith('/user/predictions')
    expect(wrapper.vm.predictions.length).toBe(0)
  })
})
