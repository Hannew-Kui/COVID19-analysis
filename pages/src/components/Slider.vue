<template>
    <vue-slider v-model="value" min="0" max="100"/>
</template>

<script lang="ts">
    export default {
        name: 'Slider'
    }
</script>

<script lang="ts" setup>
    import { ref } from 'vue';
    import VueSlider from 'vue-slider-component';
    import 'vue-slider-component/theme/default.css';

    const emit = defineEmits(['input'])
    let value = ref(0)

    function updateValue(event: any) {
        value = event.target.value
        emit('input', value)
    }
</script>