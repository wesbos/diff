import { snapdom } from "@zumer/snapdom";
import { hslToRgb } from "./utils";

const contestantEl = document.getElementById("contestant") as HTMLElement;
const targetEl = document.getElementById("target") as HTMLElement;
const compareBtn = document.getElementById("compare-btn") as HTMLButtonElement;
const pixelScoreEl = document.getElementById("pixel-score") as HTMLElement;
const contestantPreview = document.getElementById(
  "contestant-preview"
) as HTMLElement;
const targetPreview = document.getElementById("target-preview") as HTMLElement;
const diffPreview = document.getElementById("diff-preview") as HTMLElement;


function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to convert canvas to blob"));
    }, "image/png");
  });
}


interface ImageData {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

async function captureElement(element: HTMLElement): Promise<HTMLImageElement> {
  const snap = await snapdom(element);
  const img = await snap.toPng();
  return img;
}

function imageToCanvas(img: HTMLImageElement): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  // We can scale things down to lose some of the detail, or make faster comparisons.
  const scaleDownFactor = 1;
  const width = (img.naturalWidth || img.width) / scaleDownFactor;
  const height = (img.naturalHeight || img.height) / scaleDownFactor;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, width, height);
  return canvas;
}

function getImageData(canvas: HTMLCanvasElement): ImageData {
  const ctx = canvas.getContext("2d")!;
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    data: imageData.data,
    width: canvas.width,
    height: canvas.height,
  };
}

function compareImages(
  img1Data: ImageData,
  img2Data: ImageData
): { score: number; diffCanvas: HTMLCanvasElement } {
  const width = Math.max(img1Data.width, img2Data.width);
  const height = Math.max(img1Data.height, img2Data.height);
  const totalPixels = width * height;

  // First pass: calculate all pixel diffs
  const pixelDiffs = new Float32Array(totalPixels);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // Get pixel from image 1 (or transparent if out of bounds)
      let r1 = 0,
        g1 = 0,
        b1 = 0,
        a1 = 0;
      if (x < img1Data.width && y < img1Data.height) {
        const i1 = (y * img1Data.width + x) * 4;
        r1 = img1Data.data[i1];
        g1 = img1Data.data[i1 + 1];
        b1 = img1Data.data[i1 + 2];
        a1 = img1Data.data[i1 + 3];
      }

      // Get pixel from image 2 (or transparent if out of bounds)
      let r2 = 0,
        g2 = 0,
        b2 = 0,
        a2 = 0;
      if (x < img2Data.width && y < img2Data.height) {
        const i2 = (y * img2Data.width + x) * 4;
        r2 = img2Data.data[i2];
        g2 = img2Data.data[i2 + 1];
        b2 = img2Data.data[i2 + 2];
        a2 = img2Data.data[i2 + 3];
      }

      // Calculate difference
      const rDiff = Math.abs(r1 - r2);
      const gDiff = Math.abs(g1 - g2);
      const bDiff = Math.abs(b1 - b2);
      const aDiff = Math.abs(a1 - a2);

      // Weighted difference (human eye is more sensitive to green)
      // Weights: 0.299 + 0.587 + 0.114 + 0.1 = 1.1, so divide by 255 * 1.1 to normalize to [0, 1]
      // Weight alpha very low so shadows are ignored
      const pixelDiff =
        (rDiff * 0.299 + gDiff * 0.587 + bDiff * 0.114 + aDiff * 0.1) /
        (255 * 1.1);

      pixelDiffs[idx] = pixelDiff;
    }
  }

  // Calc total diff
  let totalDiff = 0;

  for (let i = 0; i < totalPixels; i++) {
    const diff = pixelDiffs[i];
    // Ignore very small diffs - close colors or alpha differences
    if(diff && diff < 0.005) {
      console.log(diff);
      continue;
    }
    totalDiff += diff;
  }

  // Third pass: paint the diff canvas
  const diffCanvas = document.createElement("canvas");
  diffCanvas.width = width;
  diffCanvas.height = height;
  const diffCtx = diffCanvas.getContext("2d")!;
  const diffImageData = diffCtx.createImageData(width, height);

  for (let i = 0; i < totalPixels; i++) {
    const pixelDiff = pixelDiffs[i];
    const rgbaIdx = i * 4;

    if (pixelDiff < 0.01) {
      // Nearly identical - leave transparent
      continue;
    }

    const hsl = hslToRgb((1 - pixelDiff) * 0.7, 1, 0.5);
    diffImageData.data[rgbaIdx] = hsl[0];
    diffImageData.data[rgbaIdx + 1] = hsl[1];
    diffImageData.data[rgbaIdx + 2] = hsl[2];
    diffImageData.data[rgbaIdx + 3] = 255;
  }

  diffCtx.putImageData(diffImageData, 0, 0);

  // Calculate similarity score (0-100)
  if (totalPixels === 0) {
    return { score: 0, diffCanvas };
  }

  const avgDiff = totalDiff / totalPixels;
  const score = Math.max(0, Math.min(100, (1 - avgDiff) * 100));

  return { score, diffCanvas };
}

function updateScoreDisplay(element: HTMLElement, score: number) {
  element.textContent = score.toFixed(1) + "%";
  element.classList.remove("high", "medium", "low");

  if (score >= 80) {
    element.classList.add("high");
  } else if (score >= 50) {
    element.classList.add("medium");
  } else {
    element.classList.add("low");
  }
}


function clearPreview(element: HTMLElement) {
  element.innerHTML = "";
}

let isComparing = false;

async function compare() {
  // Prevent overlapping comparisons
  if (isComparing) return;
  isComparing = true;

  try {
    // Capture both elements using snapdom
    const [contestantImg, targetImg] = await Promise.all([
      captureElement(contestantEl),
      captureElement(targetEl),
    ]);

    // Convert to canvas for pixel comparison
    const contestantCanvas = imageToCanvas(contestantImg);
    const targetCanvas = imageToCanvas(targetImg);

    // Get image data
    const contestantData = getImageData(contestantCanvas);
    const targetData = getImageData(targetCanvas);

    // Compare with pixel diff
    const { score: pixelScore, diffCanvas } = compareImages(
      contestantData,
      targetData
    );


    // Compare with AI Vision (if model is ready)
    // const aiScore = await calculateAISimilarity(contestantCanvas, targetCanvas);

    // Update UI
    clearPreview(contestantPreview);
    clearPreview(targetPreview);
    clearPreview(diffPreview);

    contestantPreview.appendChild(contestantImg.cloneNode());
    targetPreview.appendChild(targetImg.cloneNode());
    diffPreview.appendChild(diffCanvas);

    // Update all scores
    updateScoreDisplay(pixelScoreEl, pixelScore);

    // if (aiScore !== null) {
    //   updateScoreDisplay(aiScoreEl, aiScore);
    // }
  } catch (error) {
    console.error("Comparison failed:", error);
  } finally {
    isComparing = false;
  }
}

// Event listeners
compareBtn.addEventListener("click", compare);

// Run comparison every 1 second
setInterval(compare, 400);

// Run initial comparison after a short delay to ensure styles are loaded
setTimeout(compare, 500);
