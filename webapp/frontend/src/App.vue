<template>
  <div id="app">
    <TimeSeriesVisualization
      :k="this.k"
      :tau="this.tau"
      :mX_past="this.mX_past"
      :mX_future="this.mX_future"
      :radii="{
        past: 7, // Radius for past covariates
        actual: 12, // Radius for actual series
        forecast: 12, // Radius for forecasted series
        future: 7, // Radius for future covariates
      }"
      :colors="generatedFixedColors"
      :border="1"
      :timestepSize="55"
    />
  </div>
</template>

<script>
// Import the component, assuming local registration
import TimeSeriesVisualization from "@/components/TimeSeriesVisualization.vue";

function interpolateColor(color1, color2, factor) {
  if (arguments.length < 3) {
    factor = 0.5;
  }
  var result = color1.slice();
  for (var i = 0; i < 3; i++) {
    result[i] = Math.round(result[i] + factor * (color2[i] - color1[i]));
  }
  return result;
}

function colorToHex(color) {
  return (
    "#" +
    color
      .map((x) => {
        const hex = x.toString(16);
        return hex.length === 1 ? "0" + hex : hex;
      })
      .join("")
  );
}

function generateRandomColorMatrix(k, n) {
  let matrix = [];
  for (let i = 0; i < k; i++) {
    let row = [];
    for (let j = 0; j < n; j++) {
      const factor = Math.random(); // Generate a random factor for interpolation
      const red = [255, 0, 0];
      const blue = [0, 0, 255];
      const color = interpolateColor(red, blue, factor); // Interpolate between red and blue
      row.push(colorToHex(color)); // Convert RGB to Hex and push to the row
    }
    matrix.push(row);
  }
  return matrix;
}

function fillArrayWithValue(k, n, v) {
  // Create a new array of length k
  const array = new Array(k);

  // Fill each element of the array with a new array of length n, filled with value v
  for (let i = 0; i < k; i++) {
    array[i] = new Array(n).fill(v);
  }

  return array;
}

export default {
  name: "App",
  components: {
    TimeSeriesVisualization,
  },
  data() {
    return {
      k: 10,
      tau: 5,
      mX_past: 3,
      mX_future: 2,
    };
  },
  computed: {
    generatedRandomColors() {
      return {
        past: generateRandomColorMatrix(this.mX_past, this.k),
        actual: generateRandomColorMatrix(1, this.k)[0],
        forecast: generateRandomColorMatrix(1, this.tau)[0],
        future: generateRandomColorMatrix(this.mX_future, this.tau),
      };
    },
    generatedFixedColors() {
      return {
        past: fillArrayWithValue(this.mX_past, this.k, "lightblue"),
        actual: fillArrayWithValue(1, this.k, "blue")[0],
        forecast: fillArrayWithValue(1, this.tau, "red")[0],
        future: fillArrayWithValue(this.mX_future, this.tau, "pink"),
      };
    },
  },
};
</script>

<style>
body {
  margin: 0 !important; /* Remove default margin from body */
  background: linear-gradient(
    109.6deg,
    rgb(36, 45, 57) 11.2%,
    rgb(16, 37, 60) 51.2%,
    rgb(0, 0, 0) 98.6%
  );
}
#app {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh; /* Make app fill the entire height of the viewport */
  color: #fff; /* Light text color for contrast */
}

/* If you have additional global styles, they can go here */
</style>
