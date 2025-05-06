<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '../utils/axiosSetup'
import MetricsChart from '@/components/MetricsChart.vue'

const router = useRouter()
const leftImage = ref<File | null>(null)
const rightImage = ref<File | null>(null)
const leftPreview = ref<string>('')
const rightPreview = ref<string>('')
const isLoading = ref(false)
const error = ref<string>('')
const predictionResult = ref<string>('')
const unetOutputs = ref<{ left: string; right: string } | null>(null)
const showUnetOutputs = ref(false)
const viewerError = ref(false)
const dialog = ref(false)
const selectedFile = ref<string | undefined>()

const currentMetrics = ref<{
  mse?: number
  mae?: number
  chamfer?: number
} | null>(null)

const depthValues = ref<{ left: number[]; right: number[] } | null>(null)
const hoverDepth = ref<{ left: number | null; right: number | null }>({ left: null, right: null })
const hoverPosition = ref<{ left: { x: number; y: number } | null; right: { x: number; y: number } | null }>({ left: null, right: null })
const pointSize = ref(0.01)
const showStats = ref(false)

const handleLeftImageChange = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (input.files && input.files[0]) {
    leftImage.value = input.files[0]
    leftPreview.value = URL.createObjectURL(input.files[0])
  }
}

const handleRightImageChange = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (input.files && input.files[0]) {
    rightImage.value = input.files[0]
    rightPreview.value = URL.createObjectURL(input.files[0])
  }
}

const checkFileAccessibility = async (url: string): Promise<boolean> => {
  try {
    const response = await api.head(url)

    return response.status === 200
  }
  catch (err) {
    console.error(`Error checking file accessibility for ${url}:`, err)

    return false
  }
}

const handleSubmit = async () => {
  if (!leftImage.value || !rightImage.value) {
    error.value = 'Please select both left and right images'

    return
  }

  isLoading.value = true
  error.value = ''
  predictionResult.value = ''
  unetOutputs.value = null
  showUnetOutputs.value = false
  currentMetrics.value = null
  depthValues.value = null
  hoverDepth.value = { left: null, right: null }
  hoverPosition.value = { left: null, right: null }

  const formData = new FormData()

  formData.append('left_image', leftImage.value)
  formData.append('right_image', rightImage.value)
  formData.append('return_unet_outputs', 'true')
  formData.append('return_depth_values', 'true')

  try {
    const response = await api.post('/predict/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    let pointCloudUrl = response.data.point_cloud_path

    if (pointCloudUrl.match(/^[A-Z]:\\|^\/|^\\/i)) {
      const mediaIndex = pointCloudUrl.indexOf('media/')
      if (mediaIndex !== -1) {
        pointCloudUrl = pointCloudUrl.substring(mediaIndex)
      }
      else {
        console.error('Could not find media/ in path:', pointCloudUrl)
        error.value = 'Invalid file path returned by server'

        return
      }
    }

    pointCloudUrl = `http://localhost:8000/${pointCloudUrl}`
    console.log('Constructed point cloud URL:', pointCloudUrl)

    const isPointCloudAccessible = await checkFileAccessibility(pointCloudUrl)

    if (!isPointCloudAccessible) {
      const predictedUrl = pointCloudUrl.endsWith('_predicted.ply')
        ? pointCloudUrl
        : `${pointCloudUrl}_predicted.ply`

      const isPredictedAccessible = await checkFileAccessibility(predictedUrl)

      if (isPredictedAccessible) {
        predictionResult.value = predictedUrl
      }
      else {
        error.value = 'Point cloud file is not accessible. Please check server configuration.'

        return
      }
    }
    else {
      predictionResult.value = pointCloudUrl
    }

    if (response.data.unet_outputs) {
      let leftUnetUrl = response.data.unet_outputs.left
      if (leftUnetUrl.match(/^[A-Z]:\\|^\/|^\\/i)) {
        const mediaIndex = leftUnetUrl.indexOf('media/')
        if (mediaIndex !== -1)
          leftUnetUrl = leftUnetUrl.substring(mediaIndex)
      }
      leftUnetUrl = `http://localhost:8000/${leftUnetUrl}`

      let rightUnetUrl = response.data.unet_outputs.right
      if (rightUnetUrl.match(/^[A-Z]:\\|^\/|^\\/i)) {
        const mediaIndex = rightUnetUrl.indexOf('media/')
        if (mediaIndex !== -1)
          rightUnetUrl = rightUnetUrl.substring(mediaIndex)
      }
      rightUnetUrl = `http://localhost:8000/${rightUnetUrl}`

      const isLeftUnetAccessible = await checkFileAccessibility(leftUnetUrl)
      const isRightUnetAccessible = await checkFileAccessibility(rightUnetUrl)

      if (!isLeftUnetAccessible || !isRightUnetAccessible)
        console.warn('One or more UNet output files are not accessible')

      unetOutputs.value = {
        left: leftUnetUrl,
        right: rightUnetUrl,
      }

      if (response.data.depth_values) {
        depthValues.value = {
          left: response.data.depth_values.left,
          right: response.data.depth_values.right,
        }
      }
    }
    else {
      console.log('No UNet outputs in response')
    }

    if (response.data.metrics)
      currentMetrics.value = response.data.metrics
  }
  catch (err) {
    error.value = 'Error processing images. Please try again.'
    console.error('Prediction error:', err)
  }
  finally {
    isLoading.value = false
  }
}

const downloadPointCloud = async () => {
  if (!predictionResult.value)
    return

  try {
    const response = await api.get(predictionResult.value, {
      responseType: 'blob',
    })

    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')

    link.href = url
    link.setAttribute('download', 'point_cloud.ply')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }
  catch (err) {
    console.error('Error downloading point cloud:', err)
    error.value = 'Error downloading point cloud'
  }
}

const handleImageHover = (event: MouseEvent, side: 'left' | 'right') => {
  if (!depthValues.value)
    return

  const img = event.target as HTMLImageElement
  const rect = img.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  const width = img.naturalWidth
  const height = img.naturalHeight

  const pixelX = Math.floor((x / rect.width) * width)
  const pixelY = Math.floor((y / rect.height) * height)

  const index = pixelY * width + pixelX
  if (index >= 0 && index < depthValues.value[side].length) {
    hoverDepth.value[side] = depthValues.value[side][index]
    hoverPosition.value[side] = { x: event.clientX, y: event.clientY }
  }
}

const handleImageLeave = (side: 'left' | 'right') => {
  hoverDepth.value[side] = null
  hoverPosition.value[side] = null
}

const viewHistory = () => {
  router.push('/prediction-history')
}

const viewPointCloud = (predictionResult: string) => {
  selectedFile.value = predictionResult
  dialog.value = true
}
</script>

<template>
  <VCard class="pa-4">
    <VCardTitle class="d-flex justify-space-between align-center pa-4">
      <span>3D Point Cloud Generation</span>
      <VBtn
        color="primary"
        variant="outlined"
        size="small"
        @click="viewHistory"
      >
        View History
      </VBtn>
    </VCardTitle>

    <VRow>
      <!-- Left Image Selection -->
      <VCol
        cols="12"
        md="6"
      >
        <VCard>
          <VCardTitle>Left Image</VCardTitle>
          <VCardText>
            <VFileInput
              v-model="leftImage"
              accept="image/*"
              label="Select Left Image"
              prepend-icon="ri-camera-line"
              @change="handleLeftImageChange"
            />
            <VImg
              v-if="leftPreview"
              :src="leftPreview"
              cover
              class="mt-2"
            />
          </VCardText>
        </VCard>
      </VCol>

      <!-- Right Image Selection -->
      <VCol
        cols="12"
        md="6"
      >
        <VCard>
          <VCardTitle>Right Image</VCardTitle>
          <VCardText>
            <VFileInput
              v-model="rightImage"
              accept="image/*"
              label="Select Right Image"
              prepend-icon="ri-camera-line"
              @change="handleRightImageChange"
            />
            <VImg
              v-if="rightPreview"
              :src="rightPreview"
              cover
              class="mt-2"
            />
          </VCardText>
        </VCard>
      </VCol>

      <!-- Submit Button -->
      <VCol
        cols="12"
        class="text-center"
      >
        <VBtn
          color="primary"
          size="large"
          :loading="isLoading"
          :disabled="!leftImage || !rightImage"
          @click="handleSubmit"
        >
          Generate Point Cloud
        </VBtn>
      </VCol>

      <!-- Error Message -->
      <VCol
        v-if="error"
        cols="12"
      >
        <VAlert
          type="error"
          class="mb-4"
        >
          {{ error }}
        </VAlert>
      </VCol>

      <!-- Metrics Chart -->
      <VCol
        v-if="currentMetrics"
        cols="12"
        md="4"
      >
        <MetricsChart :metrics="currentMetrics" />
      </VCol>

      <!-- UNet Outputs Toggle -->
      <VCol
        v-if="unetOutputs"
        cols="12"
        class="text-center"
      >
        <VBtn
          color="secondary"
          :text="showUnetOutputs ? 'Hide UNet Outputs' : 'Show UNet Outputs'"
          @click="showUnetOutputs = !showUnetOutputs"
        />
      </VCol>

      <!-- UNet Outputs -->
      <VCol
        v-if="unetOutputs && showUnetOutputs"
        cols="12"
      >
        <VCard>
          <VCardTitle>UNet Outputs</VCardTitle>
          <VCardText>
            <VRow>
              <VCol
                cols="12"
                md="6"
              >
                <VCard>
                  <VCardTitle>Left Image UNet Output</VCardTitle>
                  <VCardText>
                    <VImg
                      :src="unetOutputs.left"
                      cover
                    />
                  </VCardText>
                </VCard>
              </VCol>
              <VCol
                cols="12"
                md="6"
              >
                <VCard>
                  <VCardTitle>Right Image UNet Output</VCardTitle>
                  <VCardText>
                    <VImg
                      :src="unetOutputs.right"
                      cover
                    />
                  </VCardText>
                </VCard>
              </VCol>
            </VRow>
          </VCardText>
        </VCard>
      </VCol>

      <!-- 3D Point Cloud Viewer -->
      <VCol
        v-if="predictionResult"
        cols="12"
      >
        <VCard>
          <VCardText>
            <div
              v-if="viewerError"
              class="text-center pa-4"
            >
              <VAlert
                type="warning"
                class="mb-4"
              >
                Unable to display the point cloud in 3D viewer. You can still download the file.
              </VAlert>
              <VBtn
                color="primary"
                prepend-icon="ri-download-line"
                @click="downloadPointCloud"
              >
                Download Point Cloud (PLY)
              </VBtn>
            </div>

            <VCol
              v-if="!viewerError"
              cols="12"
              class="text-center"
            >
              <VBtn
                color="primary"
                variant="text"
                size="small"
                class="me-2"
                @click="viewPointCloud(predictionResult)"
              >
                View Point Cloud
              </VBtn>
              <VDialog
                v-model="dialog"
                fullscreen
              >
                <VCard>
                  <VCardTitle class="d-flex justify-space-between align-center">
                    Point Cloud Viewer
                    <VBtn
                      icon
                      @click="dialog = false"
                    >
                      <VIcon icon="ri-close-line" />
                    </VBtn>
                  </VCardTitle>
                  <VCardText class="pa-0">
                    <PointCloudViewer
                      v-if="selectedFile"
                      :file-path="selectedFile"
                    />
                  </VCardText>
                </VCard>
              </VDialog>
            </VCol>
          </VCardText>
        </VCard>
      </VCol>
    </VRow>
  </VCard>
</template>

<style scoped>
.v-card {
  transition: all 0.3s ease;
}

.v-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.image-container {
  position: relative;
  display: inline-block;
}

.depth-tooltip {
  position: fixed;
  background-color: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  z-index: 1000;
  transform: translate(-50%, -100%);
  margin-top: -10px;
}
</style>
