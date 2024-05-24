<template>
  <div class="left-pannel">
    <div class="main-info">
      <div class="count-board" style="flex: 1; margin-left: 2rem;">
        <span>Total Cases</span>
        <span style="font-size: xx-large; color: black;">{{totalCases}}</span>
      </div>
      <div class="count-board" style="flex: 1; margin-right: 2rem;">
        <span>Total Death</span>
        <span style="font-size: xx-large; color: black;">{{totalDeath}}</span>
      </div>
    </div>
    <MapChart @handle-click="handleClick" style="flex: 3;"/>
  </div>
  <div class="right-pannel">
    <BarChart :data="confirmedOption" title="Daily Confirmed"/>
    <BarChart :data="deathOption" title="Daily Death"/>
    <BarChart :data="recoveredOption" title="Daily Recovered"/>
  </div>
</template>

<script lang="ts">
  import MapChart from "./components/MapChart.vue"
  import BarChart from './components/BarChart.vue';
  export default {
    name: 'App',
    components: {MapChart, BarChart},
  }
</script>

<script lang="ts" setup>
  import { ref, reactive, computed } from 'vue'

  import ConfirmedData from './assets/raw-data/confirmed.json'
  import DeathData from './assets/raw-data/death.json'
  import RecoveredData from './assets/raw-data/recovered.json'

  let targetDate = ref(0)
  let targetProvince = ref('Beijing')

  let confirmedOption = computed(() => {
    return ConfirmedData[targetProvince.value as keyof typeof ConfirmedData]
  })
  let deathOption = computed(() => {
    return DeathData[targetProvince.value as keyof typeof ConfirmedData]
  })
  let recoveredOption = computed(() => {
    return RecoveredData[targetProvince.value as keyof typeof ConfirmedData]
  })

  let totalCases = computed(() => {
    return confirmedOption.value.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
  })
  let totalDeath = computed(() => {
    return deathOption.value.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
  })

  function handleClick(str: string) {
    targetProvince.value = str
  }
</script>

<style scoped>

.left-pannel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.right-pannel {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.main-info {
  display: flex;
  flex: 1;
  background-color: #A0A0A0;

  box-shadow: 0 0 0.5rem 0.1rem #00000050;
}

.count-board {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  color: #303030;
  font-size: x-large;
  font-family: 'consolas';
  font-weight: bolder;
}

BarChart {
  flex: 1;
}

</style>
