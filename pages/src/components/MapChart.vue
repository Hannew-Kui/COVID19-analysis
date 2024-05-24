<template>
    <div ref="map" id="map"></div>
</template>

<script lang="ts">
    export default {
        name: 'MapChart',
    }
</script>

<script lang="ts" setup>
    import { onMounted, reactive, ref, computed } from 'vue';
    import * as echarts from 'echarts';

    import ConfirmedData from '../assets/raw-data/confirmed.json'
    import ChinaMap from '../assets/map/china.json';
    import Cn2En from '../assets/raw-data/name-map.json'

    let En2Cn: Record<string, string> = reactive({})
    Object.entries(Cn2En).forEach(([key, value]) => {
        En2Cn[value] = key
    })

    interface Pair {
        name: string,
        value: number,
    }
    let provinceConfirmDaily: Pair[] = reactive([])
    for (let key in ConfirmedData) {
        provinceConfirmDaily.push({name: En2Cn[key as keyof typeof En2Cn], value: ConfirmedData[key as keyof typeof ConfirmedData][0]}) // 暂时为第0天
    }

    let maxConfirmDaily = computed(() => {
        let max = 0
        for (let pair of provinceConfirmDaily) {
            max = pair.value > max ? pair.value : max;
        }
        return max
    })

    const map = ref(null);
    const emit = defineEmits(['handleClick'])

    onMounted(() => {

        // @ts-ignore
        echarts.registerMap('china', ChinaMap);
        const mapChart = echarts.init(map.value);
        mapChart.setOption({
            tooltip: {},
            visualMap: {
                left: '3%',
                bottom: '4%',
                min: 0,
                max: maxConfirmDaily.value,
                inRange: {
                    color: [
                        '#e9e9e9',
                        '#d67b97',
                        '#aa375b'
                    ]
                }
            },
            series: [
                {
                    name: 'CN COVID',
                    type: 'map',
                    map: 'china',
                    zoom: 1,
                    roam: true,
                    data: provinceConfirmDaily,
                    label: {
                        show: false
                    }
                }
            ]
        })
        mapChart.on('click', function (params) {
            if (params.name in Cn2En) {
                emit('handleClick', Cn2En[params.name as keyof typeof Cn2En])
            }
        })
        
    })
</script>

<style scoped>

#map {
    width: 100%;
    height: 100%;

    background-color: #808080;
    box-shadow: 0 0 0.5rem 0.1rem #00000050;
}

</style>