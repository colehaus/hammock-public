// We use JSDoc syntax here so we can use TypeScript
// without having to set up a whole build system for this one file

// Replaced with a generated ID by plotly when including this file from python
const plotId = '{plot_id}';

// Just to tell TS about it
// @ts-ignore
const Plotly = /** @type {any} */ (window.Plotly);

/** @typedef {{
 *   x: number,
 *   y: number,
 *   z: number,
 *   text: string,
 *   focalAnnotation: boolean,
 *   showarrow: boolean,
 *   bgcolor: string,
 *   borderpad: number,
 *   font: {color: string},
 *   xanchor: "left" | "center" | "right",
 *   yanchor: "bottom" | "middle" | "top",
 * }} Annotation
 */

/**
 * @typedef {{
 *   scene: {
 *     camera: {eye: Point3D},
 *     annotations: Array<Annotation>
 *   }
 * }} Layout
 */

/**
 * @typedef {{
 *   x: Array<number>,
 *   y: Array<number>,
 *   z: Array<number>,
 *   marker: {
 *     color: Array<string>
 *   },
 *   hovertemplate: string,
 * }} CoreTrace
 */

/**
 * Trace data containing individual text markers (i.e. representing paragraphs or sentences)
 * @typedef {{
 *   mode: "markers",
 *   type: "scatter3d",
 *   customdata: Array<{
 *     colors: Array<string>
 *   }>
 * } & CoreTrace} MarkerTrace,

/** @typedef { MarkerTrace & {text: Array<string>}} TextTrace */

/**
 * Trace data containing cluster labels
 * @typedef {{
 *   mode: "markers+text",
 *   type: "scatter3d",
 *   visible: boolean,
 *   name: string,
 * } & CoreTrace} ClusterTrace,
 */

// We cheat on the type declarations here because we initialize these ASAP
/** @type {Array<TextTrace>} */
let textTraces = /** @type {any} */ (null);
/** @type {Array<ClusterTrace>} */
let clusterTraces = /** @type {any} */ (null);

// Initialize camera
Plotly.update(
  plotId,
  {},
  {
    'scene.camera.eye': {
      x: -1.25,
      y: 1.25,
      z: 1.25,
    },
  }
);

/**
 * Rescale to [0, 1] and center around 0
 * @type {(axis: Array<number>, index: number) => number}
 */
function minmaxScale(axis, index) {
  const max = Math.max(...axis);
  const min = Math.min(...axis);
  const range = max - min;
  const mid = (max + min) / 2;
  return ((axis[index] - mid) / range) * 2;
}

/** @type {(backgroundColor: string) => boolean} */
function isLightColor(backgroundColor) {
  // Parse the background color string (assuming it's in the format "rgba(r, g, b, a)")
  const match = backgroundColor.match(/rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)/i);
  if (match) {
    const [r, g, b, a] = Array(...match)
      .slice(1)
      .map(parseFloat);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) * a;
    return luminance > 128;
  } else {
    return true;
  }
}

/**
 * @type {(x: number, power: number) => number}
 * We expect both `x` and the return value to be in [0, 1]
 */
function easeInOutPower(x, power) {
  return x < 0.5 ? 0.5 * Math.pow(2 * x, power) : 1 - 0.5 * Math.pow(-2 * x + 2, power);
}

/** @typedef {{x: number, y: number, z: number}} Point3D */

/**
 * Ease the camera in and out from `currentEye` to `targetEye` over `duration`
 * @type {
     (timestamp: number, startTime: number | null, duration: number, currentEye: Point3D, targetEye: Point3D) => void
 * }
 */
function animateCamera(timestamp, startTime, duration, currentEye, targetEye) {
  if (!startTime) startTime = timestamp;

  const elapsedTime = timestamp - startTime;
  const frac = easeInOutPower(elapsedTime / duration, 4);

  const view_update = {
    'scene.camera.eye': Object.fromEntries(
      /** @type {const} */ (['x', 'y', 'z']).map((axis) => [
        axis,
        currentEye[axis] + frac * (targetEye[axis] - currentEye[axis]),
      ])
    ),
    'scene.camera.center': Object.fromEntries(['x', 'y', 'z'].map((axis) => [axis, 0])),
  };
  Plotly.update(plotId, {}, view_update);

  if (elapsedTime < duration) {
    requestAnimationFrame((newTimestamp) =>
      animateCamera(newTimestamp, startTime, duration, currentEye, targetEye)
    );
  }
}

/** @type {(trace: TextTrace, index: MarkerIndex) => Annotation} */
function focalAnnotationFromTraceInfo(trace, index) {
  // TS has trouble with `fromEntries`
  const fromTrace =
    /** @type {Point3D & {text: string}} */
    (Object.fromEntries(/** @type {const} */ (['x', 'y', 'z', 'text']).map((prop) => [prop, trace[prop][index]])));
  return {
    ...fromTrace,
    // We want some trace of what type of annotation this is so we
    // can filter out the old focal annotation before creating the new one
    focalAnnotation: true,
    showarrow: true,
    bgcolor: trace.marker.color[index],
    borderpad: 4,
    font: {
      color: isLightColor(trace.marker.color[index]) ? 'black' : 'white',
    },
    xanchor: 'left',
    yanchor: 'bottom',
  };
}

/** @param {ColorIndex} colorIndex */
function setTraceVisibility(colorIndex) {
  clusterTraces.forEach((trace, i) => {
    trace.visible = i === colorIndex;
  });
  Plotly.react(
    plotId,
    textTraces.concat(/** @type {any} */ (clusterTraces)),
    /** @type any */ (document.getElementById(plotId)).layout
  );
}

/**
 * Newtype distinguishing indices into alternative color schemes from arbitrary numbers
 * @typedef {number & {_brand: "ColorIndex"}} ColorIndex
 */

/** @param {ColorIndex} colorIndex */
function swapColors(colorIndex) {
  textTraces.forEach((trace, i) => {
    // Colors are strings in format like "rgba(255, 255, 255, 0.1)"
    const colors = trace.customdata.map((x) => x.colors[colorIndex]);
    Plotly.restyle(
      plotId,
      {
        'marker.color': [colors],
        'line.color': [colors],
      },
      [i]
    );
  });
}

/** @type {[() => ColorIndex, () => ColorIndex]} */
const crementColorIndex = (() => {
  let currentColorIndex = 0;
  return [
    () => {
      // Can't float this out because `textTraces` isn't defined by the time the constructor runs
      // We assume all text traces and all points in each text trace have the same number of colors
      // (should be true by construction)
      const numColorings = textTraces[0].customdata[0].colors.length;
      currentColorIndex = (currentColorIndex + 1) % numColorings;
      return /** @type {ColorIndex} */ (currentColorIndex);
    },
    () => {
      const numColorings = textTraces[0].customdata[0].colors.length;
      currentColorIndex = (currentColorIndex - 1 + numColorings) % numColorings;
      return /** @type {ColorIndex} */ (currentColorIndex);
    },
  ];
})();
const [incrementColorIndex, decrementColorIndex] = crementColorIndex;

/**
 * @typedef {number & {_brand: "MarkerIndex"}} MarkerIndex
 * @typedef {number & {_brand: "TextTraceIndex"}} TextTraceIndex
 */

/**
 * Adds annotation for chosen marker and starts camera animation zooming to marker
 * @type {(textTraceIndex: TextTraceIndex, markerIndex: MarkerIndex) => void}
 */
const focusOnMarker = (() => {
  /** @type {number | null} */
  let timeoutHandle = null;
  return (textTraceIndex, markerIndex) => {
    const trace = textTraces[textTraceIndex];
    const currentLayout = /** @type Layout */ (/** @type any */ (document.getElementById(plotId)).layout);

    const targetEye = /** @type {Point3D} */ (
      Object.fromEntries(
        /** @type {const} */ (['x', 'y', 'z']).map((axis) => [axis, minmaxScale(trace[axis], markerIndex)])
      )
    );
    const currentEye = currentLayout.scene.camera.eye;
    const distance = Math.sqrt(
      /** @type {const} */ (['x', 'y', 'z'])
        .map((axis) => Math.pow(targetEye[axis] - currentEye[axis], 2))
        .reduce((acc, el) => acc + el)
    );
    // `cancelAnimationFrame` actually handles `null` gracefully, but TS doesn't know that
    cancelAnimationFrame(/** @type {number} */ (timeoutHandle));
    const duration = Math.min(4000, Math.max(2000, Math.sqrt(distance) * 3000));
    timeoutHandle = requestAnimationFrame((timestamp) =>
      animateCamera(timestamp, null, duration, currentEye, targetEye)
    );

    const updatedAnnotations = (currentLayout.scene.annotations ?? [])
      .filter((x) => !x.focalAnnotation)
      .concat([focalAnnotationFromTraceInfo(trace, markerIndex)]);
    Plotly.react(plotId, textTraces.concat(/** @type {any} */ (clusterTraces)), {
      ...currentLayout,
      scene: {
        ...currentLayout.scene,
        annotations: updatedAnnotations,
      },
    });
  };
})();

/** @type {[() => [TextTraceIndex, MarkerIndex], () => [TextTraceIndex, MarkerIndex]]} */
const crementMarkerIndex = (() => {
  let currentTraceIndex = 0;
  let currentMarkerIndex = 0;
  return [
    () => {
      if (currentMarkerIndex === textTraces[currentTraceIndex].x.length - 1) {
        currentMarkerIndex = 0;
        if (currentTraceIndex === textTraces.length - 1) {
          currentTraceIndex = 0;
        } else {
          currentTraceIndex += 1;
        }
      } else {
        currentMarkerIndex += 1;
      }
      return /** @type {[TextTraceIndex, MarkerIndex]} */ ([currentTraceIndex, currentMarkerIndex]);
    },
    () => {
      if (currentMarkerIndex === 0) {
        if (currentTraceIndex === 0) {
          currentTraceIndex = textTraces.length - 1;
        } else {
          currentTraceIndex -= 1;
        }
        currentMarkerIndex = textTraces[currentTraceIndex].x.length - 1;
      } else {
        currentMarkerIndex -= 1;
      }
      return /** @type {[TextTraceIndex, MarkerIndex]} */ ([currentTraceIndex, currentMarkerIndex]);
    },
  ];
})();
const [incrementMarkerIndex, decrementMarkerIndex] = crementMarkerIndex;

/** @type {(html: string) => DocumentFragment} */
function elementFromHtml(html) {
  const template = document.createElement('template');
  template.innerHTML = html;
  return template.content;
}

/** @type {<T>(array: Array<T>, predicate: (el: T) => boolean) => Array<number>} */
function findIndices(array, predicate) {
  return array.map((el, i) => (predicate(el) ? i : -1)).filter((x) => x !== -1);
}

function searchHandler(event) {
  event.preventDefault();
  const text = /** @type HTMLInputElement */ (document.getElementById('marker-search-input')).value;
  const textTraceIndices = findIndices(textTraces, (trace) => trace.text.some((str) => str.includes(text)));
  if (textTraceIndices.length === 0) {
    alert(`Found 0 markers found matching "${text}"`);
    return;
  }
  const indexPairs = /** @type {Array<[TextTraceIndex, MarkerIndex]>} */ (
    textTraceIndices.flatMap((textTraceIndex) =>
      findIndices(textTraces[textTraceIndex].text, (str) => str.includes(text)).map((markerIndex) => [
        textTraceIndex,
        markerIndex,
      ])
    )
  );
  if (indexPairs.length === 1) {
    focusOnMarker(...indexPairs[0]);
  } else if (indexPairs.length == 0) {
    alert(`Found 0 markers matching "${text}"`);
  } else {
    const matchingMarkers = indexPairs.map(
      ([textTraceIndex, markerIndex]) => textTraces[textTraceIndex].text[markerIndex]
    );
    alert(`Found ${indexPairs.length} markers matching "${text}":\n${matchingMarkers.join('\n')}`);
  }
}

function addSearchHandler() {
  const button = /** @type HTMLElement */ (document.getElementById('marker-search-submit'));
  button.addEventListener('click', searchHandler);
  const input = /** @type HTMLElement */ (document.getElementById('marker-search-input'));
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      searchHandler(event);
    }
  });
}

document.addEventListener('touchstart', function (event) {
  focusOnMarker(...incrementMarkerIndex());
});
document.addEventListener('keydown', function (event) {
  if (event.key === 'ArrowRight') {
    focusOnMarker(...incrementMarkerIndex());
  } else if (event.key === 'ArrowLeft') {
    focusOnMarker(...decrementMarkerIndex());
  } else if (event.key === 'ArrowDown') {
    const newIndex = decrementColorIndex();
    swapColors(newIndex);
    setTraceVisibility(newIndex);
  } else if (event.key === 'ArrowUp') {
    const newIndex = incrementColorIndex();
    swapColors(newIndex);
    setTraceVisibility(newIndex);
  }
});
document.addEventListener('DOMContentLoaded', function () {
  const plot = /** @type any */ (document.getElementById(plotId));
  // @ts-ignore
  textTraces = plot.data.filter((trace) => !('name' in trace));
  // @ts-ignore
  clusterTraces = plot.data.filter((trace) => 'name' in trace && trace.name.startsWith('cluster_markers_'));
  console.log(plotId);
  // Note that we're sort of cheating here by typing the `textTraces` as `Array<TextTrace>` and
  // not `Array<MarkerTrace | TextTrace>`
  if ('text' in textTraces[0]) {
    const markerSearchHtml =
      '<label for="marker-search-input">Marker Text: </label><input type="text" id="marker-search-input" name="marker-search-input"><button id="marker-search-submit">Submit</button>';
    document.body.insertBefore(elementFromHtml(markerSearchHtml), document.body.firstChild);
    addSearchHandler();
  }
});
