<template>
  <VCard class="metrics-chart pa-4">
    <VCardTitle>Prediction Metrics</VCardTitle>
    <VCardText>
      <div class="chart-container">
        <Bar
          v-if="chartData"
          :data="chartData"
          :options="chartOptions"
        />
      </div>
    </VCardText>
  </VCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
)

const props = defineProps<{
  metrics: {
    mse?: number
    mae?: number
    chamfer?: number
  }
}>()

const chartData = computed(() => {
  if (!props.metrics) return null

  const metrics = [
    { name: 'MSE', value: props.metrics.mse },
    { name: 'MAE', value: props.metrics.mae },
    { name: 'Chamfer', value: props.metrics.chamfer }
  ].filter(metric => metric.value !== undefined)

  return {
    labels: metrics.map(m => m.name),
    datasets: [
      {
        label: 'Metric Value',
        data: metrics.map(m => m.value as number),
        backgroundColor: [
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 99, 132, 0.5)',
          'rgba(75, 192, 192, 0.5)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(75, 192, 192, 1)'
        ],
        borderWidth: 1
      }
    ]
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false
    },
    title: {
      display: true,
      text: 'Prediction Metrics Comparison'
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      title: {
        display: true,
        text: 'Value'
      }
    }
  }
}
</script>

<style scoped>
.metrics-chart {
  height: 100%;
}

.chart-container {
  height: 300px;
  position: relative;
}
</style> 