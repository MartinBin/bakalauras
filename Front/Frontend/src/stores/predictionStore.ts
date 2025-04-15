import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface Prediction {
  id: string
  timestamp: number
  leftImage: string
  rightImage: string
  predictedPointCloud: string
  metrics?: {
    mse: number
    mae: number
    chamfer: number
  }
}

export const usePredictionStore = defineStore('prediction', () => {
  const predictions = ref<Prediction[]>([])
  const isLoading = ref(false)

  // Load predictions from localStorage on store initialization
  const loadPredictions = () => {
    const savedPredictions = localStorage.getItem('predictions')
    if (savedPredictions) {
      predictions.value = JSON.parse(savedPredictions)
    }
  }

  // Save predictions to localStorage
  const savePredictions = () => {
    localStorage.setItem('predictions', JSON.stringify(predictions.value))
  }

  // Add a new prediction
  const addPrediction = (prediction: Omit<Prediction, 'id' | 'timestamp'>) => {
    const newPrediction: Prediction = {
      ...prediction,
      id: crypto.randomUUID(),
      timestamp: Date.now()
    }
    predictions.value.unshift(newPrediction)
    savePredictions()
  }

  // Remove a prediction
  const removePrediction = (id: string) => {
    predictions.value = predictions.value.filter(p => p.id !== id)
    savePredictions()
  }

  // Clear all predictions
  const clearPredictions = () => {
    predictions.value = []
    savePredictions()
  }

  // Get prediction by ID
  const getPrediction = (id: string) => {
    return predictions.value.find(p => p.id === id)
  }

  // Initialize store
  loadPredictions()

  return {
    predictions,
    isLoading,
    addPrediction,
    removePrediction,
    clearPredictions,
    getPrediction
  }
}) 