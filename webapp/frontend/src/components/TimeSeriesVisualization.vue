<template>
  <div class="visualization-container">
    <!-- Past covariate rows, aligned with the actual series -->
    <div class="row" v-for="(m, mIndex) in mX_past" :key="`past-${mIndex}`">
      <div
        v-for="i in k"
        :key="`past-${mIndex}-${i}`"
        class="circle-container"
        :style="circleContainerStyle(radii.past)"
      >
        <div
          class="circle past"
          :style="circleStyle(radii.past, colors.past[mIndex][i - 1])"
        ></div>
      </div>
    </div>
    <!-- Actual and forecasted series row -->
    <div class="row actual-forecasted" :key="`actual-forecasted`">
      <div
        v-for="i in k"
        :key="`actual-${i}`"
        class="circle-container"
        :style="circleContainerStyle(radii.actual)"
      >
        <div
          class="circle actual"
          :style="circleStyle(radii.actual, colors.actual[i - 1])"
        ></div>
      </div>
      <div
        v-for="i in tau"
        :key="`forecast-${i}`"
        class="circle-container"
        :style="circleContainerStyle(radii.forecast)"
      >
        <div
          class="circle forecasted"
          :style="circleStyle(radii.forecast, colors.forecast[i - 1])"
          @mouseover="handleHover(i)"
        ></div>
      </div>
    </div>
    <!-- Future covariate rows, aligned with the forecasted series -->
    <div class="row" v-for="(m, mIndex) in mX_future" :key="`future-${mIndex}`">
      <div :style="{ width: `${timestepSize * k}px` }"></div>
      <!-- Offset for future covariates -->
      <div
        v-for="i in tau"
        :key="`future-${mIndex}-${i}`"
        class="circle-container"
        :style="circleContainerStyle(radii.future)"
      >
        <div
          class="circle future"
          :style="circleStyle(radii.future, colors.future[mIndex][i - 1])"
        ></div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    k: Number,
    tau: Number,
    mX_past: Number,
    mX_future: Number,
    radii: Object,
    colors: Object, // This now expects an object with vectors/matrices for each category
    border: Number,
    timestepSize: Number,
  },
  methods: {
    circleStyle(radius, color) {
      return {
        width: `${radius * 2}px`,
        height: `${radius * 2}px`,
        borderRadius: "50%",
        backgroundColor: color,
        border: `${this.border}px solid`,
      };
    },
    circleContainerStyle(radius) {
      return {
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: `0 ${
          (this.timestepSize - radius * 2 - this.border * 2) / 2
        }px`,
      };
    },
    handleHover(circleId) {
      console.log(`Hovered over circle: ${circleId}`);
    },
  },
};
</script>

<style scoped>
.visualization-container {
  display: flex;
  flex-direction: column;
}

.row {
  display: flex;
  margin-bottom: 10px;
}

.actual-forecasted {
  margin-top: 10px;
  margin-bottom: 20px;
}

.circle-container {
  display: flex;
}

.circle {
  display: flex;
  justify-content: center;
  align-items: center;
  border-color: azure;
}
</style>
