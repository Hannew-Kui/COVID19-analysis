<template>
    <div ref="bar" id="bar"></div>
</template>

<script lang="ts">
    export default {
        name: 'BarChart',
    }
</script>

<script lang="ts" setup>
    import { onMounted, ref, reactive, computed, watch } from 'vue';
    import * as echarts from 'echarts';

    let args = defineProps(['data', 'title'])
    let data = computed(() => {
        return args.data
    })
    const bar = ref(null)
    let barChart: echarts.ECharts | null = null

    let optionDefault = reactive({
        title: {
            text: args.title
        },
        xAxis: {
            type: 'category',
            splitLine: {
                show: false
            },
            axisLabel: {
                color: '#333'
            }
        },
        yAxis: {
            type: 'value',
            splitLine: {
                show: false
            },
            axisLabel: {
                color: '#333'
            }
        },
        series: [
            {
                data: data,
                // data: [0, 1, 12, 5],
                type: 'bar',
                itemStyle: {
                    color: '#000'
                }
            }
        ]
    })

    onMounted(() => {
        if(args.data != undefined) {
            barChart = echarts.init(bar.value)
            barChart.setOption(optionDefault)
        }
    })

    watch(args, () => {
        if(args.data != undefined && barChart != null) {
            barChart.setOption(optionDefault)
        }
    })

</script>

<style scoped>

#bar {
    width: 100%;
    height: 100%;

    background-color: #808080;
    box-shadow: 0 0 0.5rem 0.1rem #00000050;

    padding: 1rem;
}

</style>