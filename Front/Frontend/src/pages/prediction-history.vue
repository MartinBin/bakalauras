<script setup lang="ts">
import { onMounted, ref } from 'vue'
import api from '../utils/axiosSetup'

interface Prediction {
  id: string
  timestamp: number
  leftImage?: string
  rightImage?: string
  predictedPointCloud?: string
  metrics: {
    variance: number
    std_dev: number
  } | null
}

const dialog = ref(false)
const selectedFile = ref<string | undefined>()

const predictions = ref<Prediction[]>([])

onMounted(async () => {
  const response = await api.get('/user/predictions/')

  predictions.value = response.data.map((item: any) => ({
    id: item.id,
    timestamp: new Date(item.created_at).getTime(),
    leftImage: `http://localhost:8000/${item.metadata.left_image_path}`,
    rightImage: `http://localhost:8000/${item.metadata.right_image_path}`,
    predictedPointCloud: item.point_cloud_path ? `/${item.point_cloud_path.replace(/\\/g, '/')}` : '',
    metrics: item.metrics || null,
  }))
})

const formatDate = (timestamp: number) => {
  return new Date(timestamp).toLocaleString()
}

const viewPointCloud = (prediction: Prediction) => {
  selectedFile.value = 'http://localhost:8000' + prediction.predictedPointCloud
  dialog.value = true
}

const removePrediction = async (id: string) => {
  await api.delete(`/user/predictions/${id}/`)
  predictions.value = predictions.value.filter(p => p.id !== id)
}

const clearHistory = async () => {
  await api.delete('/user/predictions')
  predictions.value = []
}
</script>

<template>
  <VCard>
    <VCardTitle class="d-flex justify-space-between align-center pa-4">
      Prediction History
      <VBtn
        color="error"
        variant="outlined"
        size="small"
        @click="clearHistory"
      >
        Clear History
      </VBtn>
    </VCardTitle>

    <VCardText class="w-100">
      <VRow>
        <VCol
          cols="12"
          class="w-100"
        >
          <VTable
            v-if="predictions.length > 0"
            class="w-100"
          >
            <thead>
              <tr>
                <th>Date</th>
                <th>Left Image</th>
                <th>Right Image</th>
                <th>Metrics</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="prediction in predictions"
                :key="prediction.id"
              >
                <td>{{ formatDate(prediction.timestamp) }}</td>
                <td>
                  <VImg
                    :src="prediction.leftImage"
                    width="100"
                    height="100"
                    cover
                    class="rounded"
                  />
                </td>
                <td>
                  <VImg
                    :src="prediction.rightImage"
                    width="100"
                    height="100"
                    cover
                    class="rounded"
                  />
                </td>
                <td>
                  <div v-if="prediction.metrics && prediction.metrics.variance != null">
                    <div>Variance: {{ prediction.metrics.variance.toFixed(4) }}</div>
                    <div>Standard deviation: {{ prediction.metrics.std_dev.toFixed(4) }}</div>
                  </div>
                  <div v-else>No metrics available</div>
                </td>
                <td>
                  <VBtn
                    color="primary"
                    variant="text"
                    size="small"
                    class="me-2"
                    @click="viewPointCloud(prediction)"
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
                  <VBtn
                    color="error"
                    variant="text"
                    size="small"
                    @click="removePrediction(prediction.id)"
                  >
                    Delete
                  </VBtn>
                </td>
              </tr>
            </tbody>
          </VTable>
          <VAlert
            v-else
            type="info"
            class="ma-4"
          >
            No prediction history available
          </VAlert>
        </VCol>
      </VRow>
    </VCardText>
  </VCard>
</template>

<style scoped>
.point-cloud-container {
  width: 100%;
  height: 500px;
  background-color: #f5f5f5;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
