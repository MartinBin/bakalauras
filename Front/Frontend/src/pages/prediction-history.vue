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

    <VCardText>
      <VTable v-if="predictions.length > 0">
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
          <tr v-for="prediction in predictions" :key="prediction.id">
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
              <div v-if="prediction.metrics">
                <div>MSE: {{ prediction.metrics.mse.toFixed(4) }}</div>
                <div>MAE: {{ prediction.metrics.mae.toFixed(4) }}</div>
                <div>Chamfer: {{ prediction.metrics.chamfer.toFixed(4) }}</div>
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
    </VCardText>

    <!-- Point Cloud Viewer Dialog -->
    <VDialog
      v-model="showPointCloudViewer"
      max-width="800"
    >
      <VCard>
        <VCardTitle class="d-flex justify-space-between align-center pa-4">
          Point Cloud Viewer
          <VBtn
            icon
            variant="text"
            @click="showPointCloudViewer = false"
          >
            <VIcon>ri-close-line</VIcon>
          </VBtn>
        </VCardTitle>
        <VCardText>
          <div v-if="selectedPrediction" class="point-cloud-container">
            <!-- Add your point cloud viewer component here -->
            <img
              :src="selectedPrediction.predictedPointCloud"
              alt="Predicted Point Cloud"
              class="w-100"
            />
          </div>
        </VCardText>
      </VCard>
    </VDialog>
  </VCard>
</template>

<script setup lang="ts">
    import { ref } from 'vue'
    import { usePredictionStore } from '@/stores/predictionStore.ts'
    import type { Prediction } from '@/stores/predictionStore.ts'

    const predictionStore = usePredictionStore()
    const showPointCloudViewer = ref(false)
    const selectedPrediction = ref<Prediction | null>(null)

    const predictions = predictionStore.predictions

    const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString()
    }

    const viewPointCloud = (prediction: Prediction) => {
    selectedPrediction.value = prediction
    showPointCloudViewer.value = true
    }

    const removePrediction = (id: string) => {
    predictionStore.removePrediction(id)
    }

    const clearHistory = () => {
    predictionStore.clearPredictions()
    }
</script>

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