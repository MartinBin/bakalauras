<script lang="ts" setup>
import { nextTick, onMounted, ref } from 'vue'
import * as THREE from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

const props = defineProps<{ filePath: string }>()

const container = ref<HTMLElement | null>(null)

let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let pointCloud: THREE.Points | null = null
let controls: OrbitControls

onMounted(async () => {
  await nextTick()

  if (!container.value) return

  const { clientWidth, clientHeight } = container.value

  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x050505)
  scene.fog = new THREE.Fog(0x050505, 2000, 3500)

  camera = new THREE.PerspectiveCamera(45, clientWidth / clientHeight, 0.1, 1000)
  camera.position.set(0, 0, 5)

  renderer = new THREE.WebGLRenderer()
  renderer.setSize(clientWidth, clientHeight)
  renderer.setPixelRatio(window.devicePixelRatio)
  container.value.appendChild(renderer.domElement)

  const resizeObserver = new ResizeObserver(() => {
    if (!container.value) return
    const { clientWidth, clientHeight } = container.value
    renderer.setSize(clientWidth, clientHeight)
    camera.aspect = clientWidth / clientHeight
    camera.updateProjectionMatrix()
  })
  resizeObserver.observe(container.value)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  controls.target.set(0, 0, 0)

  const loader = new PLYLoader()

  loader.load(props.filePath, geometry => {
    geometry.computeVertexNormals()

    const positionAttr = geometry.getAttribute('position')
    const colorAttr = new THREE.Float32BufferAttribute(positionAttr.count * 3, 3)

    const pointA = new THREE.Vector3(0, 0, -1)
    const pointB = new THREE.Vector3(0, 0, 1)
    const direction = new THREE.Vector3().subVectors(pointB, pointA)
    const length = direction.length()

    direction.normalize()

    for (let i = 0; i < positionAttr.count; i++) {
      const x = positionAttr.getX(i)
      const y = positionAttr.getY(i)
      const z = positionAttr.getZ(i)

      const point = new THREE.Vector3(x, y, z)

      const relative = new THREE.Vector3().subVectors(point, pointA)
      const t = THREE.MathUtils.clamp(relative.dot(direction) / length, 0, 1)

      const color = new THREE.Color()

      color.setHSL(t, 1.0, 0.5)

      colorAttr.setXYZ(i, color.r, color.g, color.b)
    }

    geometry.setAttribute('color', colorAttr)

    const material = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
    })

    pointCloud = new THREE.Points(geometry, material)
    scene.add(pointCloud)
    renderScene()
  })

  window.addEventListener('resize', onWindowResize)
})

function renderScene() {
  requestAnimationFrame(renderScene)
  controls.update()
  renderer.render(scene, camera)
}

function onWindowResize() {
  if (!container.value) return
  const { clientWidth, clientHeight } = container.value

  renderer.setSize(clientWidth, clientHeight)
  camera.aspect = clientWidth / clientHeight
  camera.updateProjectionMatrix()
}
</script>

<template>
  <div
    ref="container"
    class="viewer-container"
  />
</template>

<style scoped>
.viewer-container {
  width: 100%;
  height: 100%;
  overflow: hidden;
}
</style>
