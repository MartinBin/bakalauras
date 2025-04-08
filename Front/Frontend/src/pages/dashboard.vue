<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue'
import axios from 'axios'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader'

const leftImage = ref<File | null>(null)
const rightImage = ref<File | null>(null)
const leftPreview = ref<string>('')
const rightPreview = ref<string>('')
const isLoading = ref(false)
const error = ref<string>('')
const predictionResult = ref<string>('')
const unetOutputs = ref<{left: string, right: string} | null>(null)
const showUnetOutputs = ref(false)
const viewerContainer = ref<HTMLElement | null>(null)
const scene = ref<THREE.Scene | null>(null)
const camera = ref<THREE.PerspectiveCamera | null>(null)
const renderer = ref<THREE.WebGLRenderer | null>(null)
const controls = ref<OrbitControls | null>(null)
const pointCloud = ref<THREE.Points | null>(null)
const animationFrameId = ref<number | null>(null)

// Initialize 3D viewer
const initViewer = () => {
  if (!viewerContainer.value) return
  
  // Create scene
  scene.value = new THREE.Scene()
  scene.value.background = new THREE.Color(0xf0f0f0)
  
  // Create camera
  camera.value = new THREE.PerspectiveCamera(
    75,
    viewerContainer.value.clientWidth / viewerContainer.value.clientHeight,
    0.1,
    1000
  )
  camera.value.position.z = 5
  
  // Create renderer
  renderer.value = new THREE.WebGLRenderer({ antialias: true })
  renderer.value.setSize(viewerContainer.value.clientWidth, viewerContainer.value.clientHeight)
  viewerContainer.value.appendChild(renderer.value.domElement)
  
  // Add controls
  controls.value = new OrbitControls(camera.value, renderer.value.domElement)
  controls.value.enableDamping = true
  controls.value.dampingFactor = 0.05
  
  // Add lights
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5)
  scene.value.add(ambientLight)
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(1, 1, 1)
  scene.value.add(directionalLight)
  
  // Start animation loop
  animate()
}

// Animation loop
const animate = () => {
  if (!scene.value || !camera.value || !renderer.value || !controls.value) return
  
  animationFrameId.value = requestAnimationFrame(animate)
  controls.value.update()
  renderer.value.render(scene.value, camera.value)
}

// Load point cloud
const loadPointCloud = async (url: string) => {
  if (!scene.value) return
  
  // Remove existing point cloud if any
  if (pointCloud.value) {
    scene.value.remove(pointCloud.value)
    pointCloud.value = null
  }
  
  try {
    const loader = new PLYLoader()
    const geometry = await loader.loadAsync(url)
    
    // Create material
    const material = new THREE.PointsMaterial({
      size: 0.01,
      vertexColors: true,
    })
    
    // Create point cloud
    pointCloud.value = new THREE.Points(geometry, material)
    scene.value.add(pointCloud.value)
    
    // Center camera on point cloud
    const box = new THREE.Box3().setFromObject(pointCloud.value)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    
    const maxDim = Math.max(size.x, size.y, size.z)
    const fov = camera.value.fov * (Math.PI / 180)
    let cameraZ = Math.abs(maxDim / Math.tan(fov / 2))
    
    camera.value.position.set(center.x, center.y, center.z + cameraZ)
    camera.value.lookAt(center)
    controls.value.target.copy(center)
  } catch (err) {
    console.error('Error loading point cloud:', err)
    error.value = 'Error loading point cloud visualization'
  }
}

// Handle window resize
const handleResize = () => {
  if (!viewerContainer.value || !camera.value || !renderer.value) return
  
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

  const formData = new FormData()
  formData.append('left_image', leftImage.value)
  formData.append('right_image', rightImage.value)
  formData.append('return_unet_outputs', 'true')

  try {
    const response = await axios.post('http://localhost:8000/api/predict/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    
    predictionResult.value = response.data.point_cloud_path
    
    // Check if UNet outputs are available
    if (response.data.unet_outputs) {
      unetOutputs.value = response.data.unet_outputs
    }
    
    // Load point cloud in 3D viewer
    await loadPointCloud(response.data.point_cloud_path)
  } catch (err) {
    error.value = 'Error processing images. Please try again.'
    console.error('Prediction error:', err)
  } finally {
    isLoading.value = false
  }
}

const downloadPointCloud = async () => {
  if (!predictionResult.value) return
  
  try {
    const response = await axios.get(predictionResult.value, {
      responseType: 'blob'
    })
    
    const url = window.URL.createObjectURL(new Blob([response.data]))
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'point_cloud.ply')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  } catch (err) {
    console.error('Error downloading point cloud:', err)
    error.value = 'Error downloading point cloud'
  }
}

// Lifecycle hooks
onMounted(() => {
  initViewer()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
  if (animationFrameId.value) {
    cancelAnimationFrame(animationFrameId.value)
  }
  
  // Clean up Three.js resources
  if (pointCloud.value) {
    pointCloud.value.geometry.dispose()
    pointCloud.value.material.dispose()
  }
  
  if (renderer.value) {
    renderer.value.dispose()
  }
  
  if (controls.value) {
    controls.value.dispose()
  }
})
</script>

<template>
  <VCard class="pa-4">
    <VCardTitle class="text-h5 mb-4">
      Point Cloud Prediction
    </VCardTitle>

    <VRow>
      <!-- Left Image Selection -->
      <VCol cols="12" md="6">
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
              height="200"
              cover
              class="mt-2"
            />
          </VCardText>
        </VCard>
      </VCol>

      <!-- Right Image Selection -->
      <VCol cols="12" md="6">
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
              height="200"
              cover
              class="mt-2"
            />
          </VCardText>
        </VCard>
      </VCol>

      <!-- Submit Button -->
      <VCol cols="12" class="text-center">
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
      <VCol v-if="error" cols="12">
        <VAlert
          type="error"
          class="mb-4"
        >
          {{ error }}
        </VAlert>
      </VCol>

      <!-- UNet Outputs Toggle -->
      <VCol v-if="unetOutputs" cols="12" class="text-center">
        <VBtn
          color="secondary"
          :text="showUnetOutputs ? 'Hide UNet Outputs' : 'Show UNet Outputs'"
          @click="showUnetOutputs = !showUnetOutputs"
        />
      </VCol>

      <!-- UNet Outputs -->
      <VCol v-if="unetOutputs && showUnetOutputs" cols="12">
        <VCard>
          <VCardTitle>UNet Outputs</VCardTitle>
          <VCardText>
            <VRow>
              <VCol cols="12" md="6">
                <VCard>
                  <VCardTitle>Left Image UNet Output</VCardTitle>
                  <VCardText>
                    <VImg
                      :src="unetOutputs.left"
                      height="300"
                      cover
                    />
                  </VCardText>
                </VCard>
              </VCol>
              <VCol cols="12" md="6">
                <VCard>
                  <VCardTitle>Right Image UNet Output</VCardTitle>
                  <VCardText>
                    <VImg
                      :src="unetOutputs.right"
                      height="300"
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
      <VCol v-if="predictionResult" cols="12">
        <VCard>
          <VCardTitle class="d-flex justify-space-between align-center">
            <span>Generated Point Cloud</span>
            <VBtn
              color="primary"
              prepend-icon="ri-download-line"
              @click="downloadPointCloud"
            >
              Download PLY
            </VBtn>
          </VCardTitle>
          <VCardText>
            <div 
              ref="viewerContainer" 
              class="point-cloud-viewer"
            ></div>
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

.point-cloud-viewer {
  width: 100%;
  height: 500px;
  background-color: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
}
</style>
