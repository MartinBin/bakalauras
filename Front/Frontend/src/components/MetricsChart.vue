<script setup lang="ts">
import { computed } from 'vue'
import { Bar } from 'vue-chartjs'
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Title, Tooltip } from 'chart.js'

const props = defineProps<{
  metrics: {
    variance?: number
    std_dev?: number
  }
}>()

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

const chartData = computed(() => {
  if (!props.metrics) return null

  const metrics = [
    { name: 'Variance', value: props.metrics.variance },
    { name: 'Standard deviation', value: props.metrics.std_dev },
  ].filter(metric => metric.value !== undefined)

  return {
    labels: metrics.map(m => m.name),
    datasets: [
      {
        label: 'Metric Value',
        data: metrics.map(m => m.value as number),
        backgroundColor: ['rgba(54, 162, 235, 0.5)', 'rgba(255, 99, 132, 0.5)', 'rgba(75, 192, 192, 0.5)'],
        borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)', 'rgba(75, 192, 192, 1)'],
        borderWidth: 1,
      },
    ],
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false,
    },
    title: {
      display: true,
      text: 'Prediction Metrics Comparison',
    },
  },
  scales: {
    y: {
      beginAtZero: true,
      title: {
        display: true,
        text: 'Value',
      },
    },
  },
}
</script>

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

<style scoped>
.metrics-chart {
  height: 100%;
}

.chart-container {
  height: 300px;
  position: relative;
}
</style>
