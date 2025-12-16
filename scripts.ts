import { snapdom } from "@zumer/snapdom";
import { ssim } from "ssim.js";
import { pipeline, cos_sim } from "@huggingface/transformers";
import type { ImageFeatureExtractionPipeline } from "@huggingface/transformers";
import { hslToRgb } from "./utils";

const contestantEl = document.getElementById("contestant") as HTMLElement;
const targetEl = document.getElementById("target") as HTMLElement;
const compareBtn = document.getElementById("compare-btn") as HTMLButtonElement;
const pixelScoreEl = document.getElementById("pixel-score") as HTMLElement;
const ssimScoreEl = document.getElementById("ssim-score") as HTMLElement;
const aiScoreEl = document.getElementById("ai-score") as HTMLElement;
const aiStatusEl = document.getElementById("ai-status") as HTMLElement;
const contestantPreview = document.getElementById(
  "contestant-preview"
) as HTMLElement;
const targetPreview = document.getElementById("target-preview") as HTMLElement;
const diffPreview = document.getElementById("diff-preview") as HTMLElement;

// AI Vision model (loaded lazily)
let imageExtractor: ImageFeatureExtractionPipeline | null = null;
let modelLoading = false;
let modelReady = false;

async function loadModel() {
  if (modelLoading || modelReady) return;
  modelLoading = true;

  try {
    aiStatusEl.textContent = "Loading model...";
    // @ts-expect-error - pipeline returns a complex union type
    imageExtractor = await pipeline(
      "image-feature-extraction",
      "Xenova/vit-base-patch16-224-in21k"
    );
    modelReady = true;
    aiStatusEl.textContent = "";
  } catch (error) {
    console.error("Failed to load AI model:", error);
    aiStatusEl.textContent = "Model failed to load";
  } finally {
    modelLoading = false;
  }
}

// Start loading model immediately
loadModel();

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to convert canvas to blob"));
    }, "image/png");
  });
}

async function calculateAISimilarity(
  canvas1: HTMLCanvasElement,
  canvas2: HTMLCanvasElement
): Promise<number | null> {
  if (!imageExtractor || !modelReady) return null;

  try {
    // Convert canvases to blobs, then to object URLs for the model
    const [blob1, blob2] = await Promise.all([
      canvasToBlob(canvas1),
      canvasToBlob(canvas2),
    ]);

    const url1 = URL.createObjectURL(blob1);
    const url2 = URL.createObjectURL(blob2);

    try {
      // Extract features with mean pooling to get a single vector per image
      const [features1, features2] = await Promise.all([
        imageExtractor(url1, { pooling: "mean" } as any),
        imageExtractor(url2, { pooling: "mean" } as any),
      ]);

      // Get embeddings as arrays
      const embedding1 = Array.from(features1.data as Float32Array);
      const embedding2 = Array.from(features2.data as Float32Array);

      // Calculate cosine similarity
      const similarity = cos_sim(embedding1, embedding2);

      // Convert to percentage (0-100)
      return Math.max(0, similarity) * 100;
    } finally {
      URL.revokeObjectURL(url1);
      URL.revokeObjectURL(url2);
    }
  } catch (error) {
    console.error("AI similarity calculation failed:", error);
    return null;
  }
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

// // Check if a pixel is "background" (transparent or white/near-white)
// function isBackground(r: number, g: number, b: number, a: number): boolean {
//   // Transparent pixels are background
//   if (a < 10) return true;

//   // White or near-white pixels are background (threshold: 250)
//   if (r > 250 && g > 250 && b > 250) return true;

//   return false;
// }

function compareImages(
  img1Data: ImageData,
  img2Data: ImageData
): { score: number; diffCanvas: HTMLCanvasElement } {
  // Create a canvas for the diff visualization
  const width = Math.max(img1Data.width, img2Data.width);
  const height = Math.max(img1Data.height, img2Data.height);
  let highestDiff;
  let lowestDiff;
  const diffCanvas = document.createElement("canvas");
  diffCanvas.width = width;
  diffCanvas.height = height;
  const diffCtx = diffCanvas.getContext("2d")!;
  const diffImageData = diffCtx.createImageData(width, height);

  let totalDiff = 0;
  let contentPixels = 0; // Only count pixels with actual content

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;

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

      // // Check if both pixels are background - skip these entirely
      // const isBg1 = isBackground(r1, g1, b1, a1);
      // const isBg2 = isBackground(r2, g2, b2, a2);

      // if (isBg1 && isBg2) {
      //   // Both are background - show as dark in diff, don't count
      //   diffImageData.data[i] = 15;
      //   diffImageData.data[i + 1] = 15;
      //   diffImageData.data[i + 2] = 20;
      //   diffImageData.data[i + 3] = 255;
      //   continue;
      // }

      // At least one has content - this pixel counts
      contentPixels++;

      // Calculate difference
      const rDiff = Math.abs(r1 - r2);
      const gDiff = Math.abs(g1 - g2);
      const bDiff = Math.abs(b1 - b2);
      const aDiff = Math.abs(a1 - a2);

      // Weighted difference (human eye is more sensitive to green)
      // Weights: 0.299 + 0.587 + 0.114 + 0.2 = 1.2, so divide by 255 * 1.2 to normalize to [0, 1]
      const pixelDiff =
        (rDiff * 0.299 + gDiff * 0.587 + bDiff * 0.114 + aDiff * 0.1) /
        (255 * 1.1);
      // const pixelDiff = (rDiff * 0.33 + gDiff * 0.33 + bDiff * 0.33 + aDiff * 0.01) / 255;
      const alphaDiffCutoff = 100;

      // Alpha Diff less than 100 is likely a small shadow difference and should be ignored
      // if(aDiff > 0 && aDiff < alphaDiffCutoff) {
      //   // Pink Pixels are Different, but skipped from counting towards the score
      //   diffImageData.data[i] = 251;
      //   diffImageData.data[i + 1] = 0;
      //   diffImageData.data[i + 2] = 255;
      //   diffImageData.data[i + 3] = 45;
      //   continue;
      // }

      // TODO: We should check if this diff pixel is extremely small and skip it if so. We can calculate this from neighboring pixels.

      // Add to total difference
      totalDiff += pixelDiff;

      if(pixelDiff < 0.01) {
        continue;
      }

      const showScaleDiff = true;
      if (showScaleDiff) {
        highestDiff = Math.max(highestDiff || 0, pixelDiff);
        lowestDiff = Math.min(lowestDiff || Infinity, pixelDiff);
        const hsl = hslToRgb((1 - pixelDiff) * 0.7, 1, 0.5);
        diffImageData.data[i] = hsl[0];
        diffImageData.data[i + 1] = hsl[1];
        diffImageData.data[i + 2] = hsl[2];
        diffImageData.data[i + 3] = 255;
        continue;
      }

      if (pixelDiff < 0.01) {
        // Nearly identical - show as green/teal (matching content!)
        // diffImageData.data[i] = 0;
        // diffImageData.data[i + 1] = 180;
        // diffImageData.data[i + 2] = 140;
        // diffImageData.data[i + 3] = 255;
      } else if (pixelDiff < 0.1) {
        // Small difference - yellow
        diffImageData.data[i] = 255;
        diffImageData.data[i + 1] = 200;
        diffImageData.data[i + 2] = 0;
        diffImageData.data[i + 3] = Math.min(255, 100 + pixelDiff * 1550);
      } else {
        // Large difference - red/magenta
        const intensity = Math.min(255, pixelDiff * 255 * 2);
        diffImageData.data[i] = 255;
        diffImageData.data[i + 1] = 50;
        diffImageData.data[i + 2] = 100;
        diffImageData.data[i + 3] = intensity;
      }
    }
  }

  console.log({ highestDiff, lowestDiff });

  diffCtx.putImageData(diffImageData, 0, 0);

  // Calculate similarity score (0-100) based only on content pixels
  // If no content pixels found, return 0
  if (contentPixels === 0) {
    return { score: 0, diffCanvas };
  }

  const avgDiff = totalDiff / contentPixels;
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

function calculateSSIM(
  canvas1: HTMLCanvasElement,
  canvas2: HTMLCanvasElement
): number {
  // SSIM requires same dimensions, so we'll use the larger of the two
  const width = Math.max(canvas1.width, canvas2.width);
  const height = Math.max(canvas1.height, canvas2.height);

  // Create normalized canvases with same dimensions
  const normalizedCanvas1 = document.createElement("canvas");
  const normalizedCanvas2 = document.createElement("canvas");
  normalizedCanvas1.width = normalizedCanvas2.width = width;
  normalizedCanvas1.height = normalizedCanvas2.height = height;

  const nCtx1 = normalizedCanvas1.getContext("2d")!;
  const nCtx2 = normalizedCanvas2.getContext("2d")!;

  // Draw originals
  nCtx1.drawImage(canvas1, 0, 0);
  nCtx2.drawImage(canvas2, 0, 0);

  const imageData1 = nCtx1.getImageData(0, 0, width, height);
  const imageData2 = nCtx2.getImageData(0, 0, width, height);

  // // Mask out background pixels in both images
  // // Set them to the same neutral value so they don't affect SSIM
  // for (let i = 0; i < imageData1.data.length; i += 4) {
  //   const r1 = imageData1.data[i];
  //   const g1 = imageData1.data[i + 1];
  //   const b1 = imageData1.data[i + 2];
  //   const a1 = imageData1.data[i + 3];

  //   const r2 = imageData2.data[i];
  //   const g2 = imageData2.data[i + 1];
  //   const b2 = imageData2.data[i + 2];
  //   const a2 = imageData2.data[i + 3];

  //   const isBg1 = isBackground(r1, g1, b1, a1);
  //   const isBg2 = isBackground(r2, g2, b2, a2);

  //   // If both are background, set to transparent
  //   if (isBg1 && isBg2) {
  //     imageData1.data[i] = imageData2.data[i] = 128;
  //     imageData1.data[i + 1] = imageData2.data[i + 1] = 128;
  //     imageData1.data[i + 2] = imageData2.data[i + 2] = 128;
  //     imageData1.data[i + 3] = imageData2.data[i + 3] = 255;
  //   }
  // }

  // Calculate SSIM on masked images
  const result = ssim(imageData1, imageData2);

  return result.mssim * 100; // Convert to percentage
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

    // Compare with SSIM
    const ssimScore = calculateSSIM(contestantCanvas, targetCanvas);

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
    updateScoreDisplay(ssimScoreEl, ssimScore);
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
