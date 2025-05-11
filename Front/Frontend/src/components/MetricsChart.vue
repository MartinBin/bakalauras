<script setup lang="ts">
import { computed } from 'vue'
import { Bar, Doughnut } from 'vue-chartjs'
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Title, Tooltip, ArcElement } from 'chart.js'

const props = defineProps<{
  metrics: {
    variance?: number
    std_dev?: number
    depth_confidence?: number
  }
}>()

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement)

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

const confidenceOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: true,
      position: 'bottom' as const,
    },
    title: {
      display: true,
      text: 'Confidence Components',
    },
  },
}
</script>

<template>
  <div class="metrics-container">
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
  </div>
</template>

<style scoped>
.metrics-container {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.metrics-chart,
.confidence-chart {
  height: 100%;
}

.chart-container {
  height: 300px;
  position: relative;
}

.charts-row {
  display: flex;
  gap: 1rem;
}

.charts-row .chart-container {
  flex: 1;
}

.confidence-score {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 2rem;
  font-weight: bold;
  color: #4bc0c0;
}
</style>
