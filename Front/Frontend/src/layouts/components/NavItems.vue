<script lang="ts" setup>
import VerticalNavSectionTitle from '@/@layouts/components/VerticalNavSectionTitle.vue'
import VerticalNavLink from '@layouts/components/VerticalNavLink.vue'
</script>

<template>
  <VerticalNavSectionTitle :item="{ heading: '3D wound generation model' }" />
  <VerticalNavLink
    :item="{
      title: 'Generate 3D',
      to: '/dashboard',

    }"
  />

  <VerticalNavLink
    :item="{
      title: 'Prediction History',
      to: '/prediction-history',
    }"
  />
</template>
