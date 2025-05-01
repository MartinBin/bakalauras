<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { usePredictionStore } from '@/stores/predictionStore'
import { useRouter } from 'vue-router'
import MetricsChart from '@/components/MetricsChart.vue'
import api from '../utils/axiosSetup'

const router = useRouter()
const predictionStore = usePredictionStore()
const leftImage = ref<File | null>(null)
const rightImage = ref<File | null>(null)
const leftPreview = ref<string>('')
const rightPreview = ref<string>('')
const isLoading = ref(false)
const error = ref<string>('')
const predictionResult = ref<string>('')
const unetOutputs = ref<{ left: string; right: string } | null>(null)
const showUnetOutputs = ref(false)
const viewerContainer = ref<HTMLElement | null>(null)
const scene = ref<THREE.Scene | null>(null)
const camera = ref<THREE.PerspectiveCamera | null>(null)
const renderer = ref<THREE.WebGLRenderer | null>(null)
const controls = ref<OrbitControls | null>(null)
const pointCloud = ref<THREE.Points | null>(null)
const animationFrameId = ref<number | null>(null)
const viewerError = ref(false)
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
const pointCloudStats = ref<{
  pointCount: number
  boundingBox: {
    min: THREE.Vector3
    max: THREE.Vector3
    dimensions: THREE.Vector3
  }
} | null>(null)

const initViewer = () => {
  if (!viewerContainer.value)
    return

  scene.value = new THREE.Scene()
  scene.value.background = new THREE.Color(0xF0F0F0)

  camera.value = new THREE.PerspectiveCamera(
    75,
    viewerContainer.value.clientWidth / viewerContainer.value.clientHeight,
    0.1,
    1000,
  )
  camera.value.position.z = 5

  renderer.value = new THREE.WebGLRenderer({ antialias: true })
  if (renderer.value) {
    renderer.value.setSize(viewerContainer.value.clientWidth, viewerContainer.value.clientHeight)
    renderer.value.setPixelRatio(window.devicePixelRatio)
    viewerContainer.value.appendChild(renderer.value.domElement)
  }

  controls.value = new OrbitControls(camera.value, renderer.value?.domElement)
  controls.value.enableDamping = true
  controls.value.dampingFactor = 0.05

  const ambientLight = new THREE.AmbientLight(0xFFFFFF, 0.5)

  scene.value.add(ambientLight)

  const directionalLight = new THREE.DirectionalLight(0xFFFFFF, 0.8)

  directionalLight.position.set(1, 1, 1)
  scene.value.add(directionalLight)

  const gridHelper = new THREE.GridHelper(10, 10)

  scene.value.add(gridHelper)

  const axesHelper = new THREE.AxesHelper(5)

  scene.value.add(axesHelper)

  console.log('3D viewer initialized')

  animate()
}

const animate = () => {
  if (!renderer.value || !scene.value || !camera.value || !controls.value)
    return

  requestAnimationFrame(animate)
  controls.value.update()
  renderer.value.render(scene.value, camera.value)
}

const updatePointSize = () => {
  if (pointCloud.value && pointCloud.value.material) {
    if (Array.isArray(pointCloud.value.material))
      pointCloud.value.material.forEach(material => (material as THREE.PointsMaterial).size = pointSize.value)
    else
      (pointCloud.value.material as THREE.PointsMaterial).size = pointSize.value
  }
}

const loadPointCloud = async (url: string) => {
  console.log('loadPointCloud called with URL:', url)
  viewerError.value = false

  if (!scene.value || !camera.value || !controls.value) {
    console.error('Missing required Three.js components:', {
      scene: !!scene.value,
      camera: !!camera.value,
      controls: !!controls.value,
    })
    viewerError.value = true
    return
  }

  if (pointCloud.value) {
    console.log('Removing existing point cloud')
    scene.value.remove(pointCloud.value)
    pointCloud.value = null
  }

  try {
    const loader = new PLYLoader()
    const response = await fetch(url)
    if (!response.ok)
      throw new Error(`Failed to fetch PLY file: ${response.status} ${response.statusText}`)

    const blob = await response.blob()
    const objectUrl = URL.createObjectURL(blob)
    const geometry = await loader.loadAsync(objectUrl)
    URL.revokeObjectURL(objectUrl)

    const positions = geometry.attributes.position.array
    const pointCount = positions.length / 3
    const boundingBox = new THREE.Box3().setFromBufferAttribute(geometry.attributes.position as THREE.BufferAttribute)
    const dimensions = new THREE.Vector3()
    boundingBox.getSize(dimensions)

    pointCloudStats.value = {
      pointCount,
      boundingBox: {
        min: boundingBox.min.clone(),
        max: boundingBox.max.clone(),
        dimensions: dimensions.clone()
      }
    }

    const material = new THREE.PointsMaterial({
      size: pointSize.value,
      vertexColors: true,
      sizeAttenuation: true
    })

    pointCloud.value = new THREE.Points(geometry, material)
    scene.value.add(pointCloud.value)

    const box = new THREE.Box3().setFromObject(pointCloud.value)
    const center = box.getCenter(new THREE.Vector3())
    const boxSize = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(boxSize.x, boxSize.y, boxSize.z)
    const fov = camera.value.fov * (Math.PI / 180)
    const cameraZ = Math.abs(maxDim / Math.tan(fov / 2)) * 1.5

    camera.value.position.set(center.x, center.y, center.z + cameraZ)
    camera.value.lookAt(center)
    controls.value.target.copy(center)
    controls.value.maxDistance = cameraZ * 2
    controls.value.minDistance = maxDim * 0.1

    console.log('Camera positioned')
  }
  catch (err) {
    console.error('Error loading point cloud:', err)
    error.value = 'Error loading point cloud visualization'
    viewerError.value = true
  }
}

const handleResize = () => {
  if (!viewerContainer.value || !camera.value || !renderer.value)
    return

  const width = viewerContainer.value.clientWidth
  const height = viewerContainer.value.clientHeight

  camera.value.aspect = width / height
  camera.value.updateProjectionMatrix()
  renderer.value.setSize(width, height)
}

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

    initViewer()

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

    if (response.data.metrics) {
      currentMetrics.value = response.data.metrics
      
      predictionStore.addPrediction({
        leftImage: leftPreview.value,
        rightImage: rightPreview.value,
        predictedPointCloud: predictionResult.value,
        metrics: response.data.metrics
      })
    }

    await loadPointCloud(predictionResult.value)
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
  if (!depthValues.value) return

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

onMounted(() => {
  initViewer()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  if (animationFrameId.value)
    cancelAnimationFrame(animationFrameId.value)

  if (pointCloud.value) {
    if (pointCloud.value.geometry)
      pointCloud.value.geometry.dispose()

    if (pointCloud.value.material) {
      if (Array.isArray(pointCloud.value.material))
        pointCloud.value.material.forEach(material => material.dispose())
      else
        pointCloud.value.material.dispose()
    }
  }

  if (renderer.value)
    renderer.value.dispose()

  if (controls.value)
    controls.value.dispose()
})

const viewHistory = () => {
  router.push('/prediction-history')
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
              v-if="!viewerError"
              ref="viewerContainer"
              class="point-cloud-viewer"
            />
            <div
              v-else
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


            <div v-if="!viewerError" class="point-cloud-controls mt-4">
              <VRow>
                <VCol cols="12" md="6">
                  <VSlider
                    v-model="pointSize"
                    label="Point Size"
                    min="0.001"
                    max="0.1"
                    step="0.001"
                    @update:model-value="updatePointSize"
                  />
                </VCol>
                <VCol cols="12" md="6">
                  <VBtn
                    color="secondary"
                    @click="showStats = !showStats"
                  >
                    {{ showStats ? 'Hide Stats' : 'Show Stats' }}
                  </VBtn>
                </VCol>
              </VRow>

              <!-- Point Cloud Statistics -->
              <VExpansionPanels v-if="showStats && pointCloudStats" class="mt-4">
                <VExpansionPanel>
                  <VExpansionPanelTitle>Point Cloud Statistics</VExpansionPanelTitle>
                  <VExpansionPanelText>
                    <VList>
                      <VListItem>
                        <VListItemTitle>Total Points</VListItemTitle>
                        <VListItemSubtitle>{{ pointCloudStats.pointCount.toLocaleString() }}</VListItemSubtitle>
                      </VListItem>
                      <VListItem>
                        <VListItemTitle>Bounding Box Dimensions</VListItemTitle>
                        <VListItemSubtitle>
                          X: {{ pointCloudStats.boundingBox.dimensions.x.toFixed(2) }},
                          Y: {{ pointCloudStats.boundingBox.dimensions.y.toFixed(2) }},
                          Z: {{ pointCloudStats.boundingBox.dimensions.z.toFixed(2) }}
                        </VListItemSubtitle>
                      </VListItem>
                      <VListItem>
                        <VListItemTitle>Bounding Box Min</VListItemTitle>
                        <VListItemSubtitle>
                          X: {{ pointCloudStats.boundingBox.min.x.toFixed(2) }},
                          Y: {{ pointCloudStats.boundingBox.min.y.toFixed(2) }},
                          Z: {{ pointCloudStats.boundingBox.min.z.toFixed(2) }}
                        </VListItemSubtitle>
                      </VListItem>
                      <VListItem>
                        <VListItemTitle>Bounding Box Max</VListItemTitle>
                        <VListItemSubtitle>
                          X: {{ pointCloudStats.boundingBox.max.x.toFixed(2) }},
                          Y: {{ pointCloudStats.boundingBox.max.y.toFixed(2) }},
                          Z: {{ pointCloudStats.boundingBox.max.z.toFixed(2) }}
                        </VListItemSubtitle>
                      </VListItem>
                    </VList>
                  </VExpansionPanelText>
                </VExpansionPanel>
              </VExpansionPanels>
            </div>
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

.point-cloud-controls {
  background-color: rgba(255, 255, 255, 0.9);
  padding: 16px;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.point-cloud-viewer {
  width: 100%;
  height: 500px;
  background-color: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
  position: relative;
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

